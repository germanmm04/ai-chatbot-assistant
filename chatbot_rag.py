from __future__ import annotations

import os

# Evita que transformers cargue TensorFlow/Keras (conflicto con Keras 3).
os.environ.setdefault("USE_TF", "0")

import argparse
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
STOPWORDS_ES = {
    "a", "al", "algo", "algun", "alguna", "algunas", "alguno", "algunos", "ante",
    "como", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde", "e",
    "el", "ella", "ellas", "ellos", "en", "entre", "era", "eramos", "eran", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estar", "este",
    "esto", "estos", "fue", "ha", "han", "hasta", "hay", "la", "las", "le", "les",
    "lo", "los", "me", "mi", "mis", "mucho", "muy", "no", "nos", "o", "os", "para",
    "pero", "por", "porque", "que", "quien", "se", "segun", "ser", "si", "sin", "sobre",
    "su", "sus", "te", "tiene", "tienen", "tu", "un", "una", "uno", "unos", "y", "ya",
}


def load_documents(input_dir: Path):
    docs = []
    for file_path in input_dir.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        file_docs = loader.load()
        for d in file_docs:
            d.metadata["source"] = file_path.name
        docs.extend(file_docs)
    return docs


def build_or_load_vectorstore(
    docs_dir: Path,
    index_dir: Path,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if index_dir.exists():
        return FAISS.load_local(
            folder_path=str(index_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    docs = load_documents(docs_dir)
    if not docs:
        raise ValueError(
            "No se encontraron documentos en la carpeta de entrada. "
            "Agrega archivos PDF/TXT/MD y vuelve a ejecutar."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    return vectorstore


def answer_with_retrieval(vectorstore: FAISS, question: str, k: int = 4):
    if is_low_information_query(question) and not is_article_short_query(question):
        return (
            "Formula una pregunta completa para poder buscar evidencia en los documentos.",
            [],
        )

    article_diff_answer, article_diff_docs = answer_article_difference_query(
        vectorstore=vectorstore, question=question
    )
    if article_diff_answer:
        return article_diff_answer, article_diff_docs

    article_answer, article_docs = answer_article_query(vectorstore=vectorstore, question=question)
    if article_answer:
        return article_answer, article_docs

    capital_answer, capital_docs = answer_capital_query(vectorstore=vectorstore, question=question)
    if capital_answer:
        return capital_answer, capital_docs

    rights_answer, rights_docs = answer_rights_listing_query(vectorstore=vectorstore, question=question)
    if rights_answer:
        return rights_answer, rights_docs

    legal_consequence_answer, legal_consequence_docs = answer_legal_consequence_query(
        vectorstore=vectorstore, question=question
    )
    if legal_consequence_answer:
        return legal_consequence_answer, legal_consequence_docs

    # Distancia baja = mayor similitud. Tomamos mas contexto para reranking local.
    retrieved = vectorstore.similarity_search_with_score(question, k=12)
    if not retrieved:
        return "No tengo evidencia suficiente en los documentos proporcionados.", []

    best_score = retrieved[0][1]
    docs = [doc for doc, _ in retrieved]
    if best_score > 1.25:
        return "No tengo evidencia suficiente en los documentos proporcionados.", docs[:2]

    intent = question_type(question)
    candidates = build_sentence_candidates(docs=docs, question=question)
    if not candidates:
        return "No tengo evidencia suficiente en los documentos proporcionados.", docs[:2]
    if int(candidates[0]["score"]) < 3:
        return "No tengo evidencia suficiente en los documentos proporcionados.", docs[:2]

    answer, used_sources = compose_answer_by_intent(candidates=candidates, question=question, intent=intent)
    if not answer:
        return "No tengo evidencia suficiente en los documentos proporcionados.", docs[:2]

    selected_docs = select_docs_by_sources(docs=docs, sources=used_sources)
    return answer, selected_docs


def normalize_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u00ad", "")
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def canonical_token(token: str) -> str:
    t = strip_accents(token.lower())
    for suffix in ("ciones", "cion", "s", "es"):
        if len(t) > 5 and t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    return t


def question_terms(question: str) -> set[str]:
    tokens = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", question.lower())
    return {
        canonical_token(t)
        for t in tokens
        if len(t) > 2 and strip_accents(t) not in STOPWORDS_ES
    }


def is_low_information_query(question: str) -> bool:
    q = question.strip()
    if not q:
        return True
    tokens = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", q.lower())
    return len(tokens) < 3 and "?" not in q


def is_article_short_query(question: str) -> bool:
    q = strip_accents(question.lower()).strip()
    return bool(re.search(r"^articulo\s+\d+$", q))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    clean_parts = []
    for p in parts:
        p = p.strip()
        if len(p) < 45:
            continue
        if is_bad_sentence(p):
            continue
        clean_parts.append(p)
    return clean_parts


def sentence_score(sentence: str, terms: set[str]) -> int:
    sentence_tokens = {
        canonical_token(t)
        for t in re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", sentence.lower())
        if len(t) > 2
    }
    overlap = sentence_tokens.intersection(terms)
    score = len(overlap) * 3

    # Penaliza oraciones con ruido de indice/titulo.
    score -= title_noise_penalty(sentence)
    return score


def build_sentence_candidates(docs, question: str) -> List[Dict[str, object]]:
    terms = question_terms(question)
    intent = question_type(question)
    target = definition_target(question)
    target_canonical = canonical_token(target) if target else ""
    candidates: List[Dict[str, object]] = []

    for d in docs:
        source = d.metadata.get("source", "desconocido")
        text = normalize_text(d.page_content)
        for sent in split_sentences(text):
            if is_bad_sentence(sent):
                continue

            score = sentence_score(sent, terms)
            sent_norm = strip_accents(sent.lower())
            sent_can = canonical_tokenized(sent_norm)

            if intent == "definition":
                if target_canonical and target_canonical in sent_can and " es " in sent_norm:
                    score += 14
                if "se define" in sent_norm or "consiste en" in sent_norm:
                    score += 7
            elif intent == "comparative":
                if any(m in sent_norm for m in ["diferencia", "mientras", "en cambio", "a diferencia"]):
                    score += 9
            elif intent == "consequence":
                if any(m in sent_norm for m in ["consecuencia", "provoca", "causa", "genera", "implica", "por lo tanto"]):
                    score += 10
            elif intent == "integration":
                # En integracion, premiamos evidencia semantica en distintas fuentes.
                score += 2

            if score > 0:
                candidates.append(
                    {
                        "score": score,
                        "sentence": sent.strip(),
                        "source": source,
                    }
                )

    candidates.sort(key=lambda x: int(x["score"]), reverse=True)
    return dedupe_candidates(candidates, limit=30)


def canonical_tokenized(text: str) -> str:
    raw_tokens = re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", text)
    tokens = [canonical_token(t) for t in raw_tokens if len(t) > 1]
    return " ".join(tokens)


def dedupe_candidates(candidates: List[Dict[str, object]], limit: int) -> List[Dict[str, object]]:
    unique: List[Dict[str, object]] = []
    seen = set()
    for c in candidates:
        key = str(c["sentence"]).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= limit:
            break
    return unique


def is_bad_sentence(sentence: str) -> bool:
    s = sentence.strip()
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return True

    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > 0.55:
        return True

    # Filtra encabezados comunes en PDFs.
    s_norm = strip_accents(s.lower())
    bad_patterns = [
        "tomo",
        "capitulo",
        "manual practico",
        "bases fisiologicas",
        "nutricion en la infancia",
        "texto consolidado",
        "ultima modificacion",
        "don juan carlos",
        "preambulo",
        "boletin oficial del estado",
    ]
    if any(p in s_norm for p in bad_patterns) and len(s.split()) < 14:
        return True
    if any(p in s_norm for p in ["texto consolidado", "don juan carlos", "preambulo"]):
        return True

    # Evita lineas con demasiados separadores de portada/indice.
    if s.count("•") >= 1 or s.count("|") >= 1:
        return True

    # Filtra referencias bibliograficas tipo "Madrid: ...; 2004."
    if re.search(r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+:\s+.+;\s*\d{4}\.?\s*$", s):
        return True

    # Filtra lineas de referencia con anio y poco contenido semantico.
    if re.search(r"\b\d{4}\b", s) and len(s.split()) <= 12:
        return True

    # Descarta frases de recomendacion sueltas no definitorias para preguntas de concepto.
    if s.lower().startswith(("a esta edad", "se recomienda", "debe consumirse")):
        return True
    if strip_accents(s.lower()).startswith(("titulo ", "capitulo ", "seccion ")):
        return True

    return False


def title_noise_penalty(sentence: str) -> int:
    s_norm = strip_accents(sentence.lower())
    penalty = 0
    if "tomo" in s_norm or "capitulo" in s_norm:
        penalty += 3
    if "manual" in s_norm and "nutricion" in s_norm:
        penalty += 2
    return penalty


def question_type(question: str) -> str:
    q = strip_accents(question.lower())
    if any(k in q for k in ["resume", "resumen", "ideas clave", "tres ideas", "sintetiza"]):
        return "summary"
    if any(k in q for k in ["integra", "relaciona informacion", "segun dos documentos", "de dos documentos", "combina"]):
        return "integration"
    if "diferencia" in q or "compara" in q:
        return "comparative"
    if "consecuencia" in q or "que pasa" in q or "efecto" in q:
        return "consequence"
    if "que es" in q or "que significa" in q or "defin" in q:
        return "definition"
    return "general"


def parse_article_number(question: str):
    q = strip_accents(question.lower())
    num_match = re.search(r"articulo\s+(\d+)", q)
    if num_match:
        return int(num_match.group(1))

    ordinals = {
        "primer": 1,
        "primero": 1,
        "segunda": 2,
        "segundo": 2,
        "tercer": 3,
        "tercero": 3,
        "cuarto": 4,
        "quinta": 5,
        "quinto": 5,
    }
    for key, value in ordinals.items():
        if key in q and "articulo" in q:
            return value
    return None


def parse_two_article_numbers(question: str):
    q = strip_accents(question.lower())
    nums = [int(n) for n in re.findall(r"articulo\s+(\d+)", q)]
    if len(nums) >= 2:
        return nums[0], nums[1]

    m = re.search(r"articulo\s+(\d+)\s+(?:del|de)\s+(\d+)", q)
    if m:
        return int(m.group(1)), int(m.group(2))

    m2 = re.search(r"(\d+)\s+y\s+(\d+)", q)
    if m2 and "articulo" in q:
        return int(m2.group(1)), int(m2.group(2))
    return None, None


def extract_article_block(text: str, article_num: int) -> str:
    clean = normalize_text(text)
    pattern = rf"\b(?:Artículo|Articulo)\s+{article_num}(?:\.|º|ª|\.º|\.ª)?\s+"
    m = re.search(pattern, clean, flags=re.IGNORECASE)
    if not m:
        return ""

    start = m.start()
    original = clean[start:]
    next_article = re.search(
        r"\b(?:Artículo|Articulo)\s+\d+(?:\.|º|ª|\.º|\.ª)?\s+",
        original[15:],
        flags=re.IGNORECASE,
    )
    block = original if not next_article else original[: 15 + next_article.start()]
    block = block.strip()
    if len(block) > 750:
        block = block[:750].rsplit(" ", 1)[0] + "..."
    return block


def answer_article_query(vectorstore: FAISS, question: str):
    q_norm = strip_accents(question.lower())
    if "articulo" not in q_norm:
        return "", []

    article_num = parse_article_number(question)
    if not article_num:
        return "", []

    # 1) Busqueda exhaustiva en todos los chunks del indice (mas robusta para textos legales).
    all_docs = get_all_index_docs(vectorstore)
    for d in all_docs:
        block = extract_article_block(d.page_content, article_num=article_num)
        if block and is_heading_like_article(block, article_num):
            source = d.metadata.get("source", "desconocido")
            d.metadata["source"] = source
            return block, [d]

    # 2) Fallback semantico amplio.
    probe = f"Articulo {article_num}"
    docs = vectorstore.similarity_search(probe, k=40)
    for d in docs:
        block = extract_article_block(d.page_content, article_num=article_num)
        if block:
            source = d.metadata.get("source", "desconocido")
            d.metadata["source"] = source
            return block, [d]

    # 3) Fallback por linea para formatos raros de OCR/PDF.
    article_line_pattern = re.compile(
        rf"(?:Artículo|Articulo)\s+{article_num}(?:\.|º|ª|\.º|\.ª)?\s+",
        flags=re.IGNORECASE,
    )
    for d in all_docs:
        text = normalize_text(d.page_content)
        hit = article_line_pattern.search(text)
        if not hit:
            continue
        start = hit.start()
        snippet = text[start : min(len(text), start + 700)].strip()
        if snippet:
            source = d.metadata.get("source", "desconocido")
            d.metadata["source"] = source
            return snippet, [d]

    return "", []


def answer_article_difference_query(vectorstore: FAISS, question: str):
    q = strip_accents(question.lower())
    if not any(k in q for k in ["diferencia", "difiere", "compar", "distinto"]):
        return "", []
    if "articulo" not in q:
        return "", []

    a1, a2 = parse_two_article_numbers(question)
    if not a1 or not a2:
        return "", []

    text1, docs1 = answer_article_query(vectorstore, f"articulo {a1}")
    text2, docs2 = answer_article_query(vectorstore, f"articulo {a2}")
    if not text1 or not text2:
        return "", []

    s1 = first_article_sentence(text1)
    s2 = first_article_sentence(text2)
    answer = (
        f"El articulo {a1} establece: {s1} "
        f"En cambio, el articulo {a2} establece: {s2}"
    )
    return answer, merge_docs_unique(docs1, docs2)


def first_article_sentence(article_block: str) -> str:
    # Elimina cabecera "Articulo X." y devuelve la primera frase util.
    cleaned = re.sub(r"^\s*(?:Artículo|Articulo)\s+\d+(?:\.|º|ª|\.º|\.ª)?\s*", "", article_block, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[\.\!\?])\s+", cleaned)
    for p in parts:
        p = p.strip()
        if len(p) >= 35:
            return p
    return cleaned.strip()[:240]


def merge_docs_unique(docs_a, docs_b):
    merged = []
    seen = set()
    for d in [*(docs_a or []), *(docs_b or [])]:
        src = d.metadata.get("source", "desconocido")
        if src in seen:
            continue
        seen.add(src)
        merged.append(d)
    return merged


def answer_capital_query(vectorstore: FAISS, question: str):
    q = strip_accents(question.lower())
    if "capital" not in q:
        return "", []
    if "estado" not in q and "espana" not in q:
        return "", []

    docs = vectorstore.similarity_search("capital del Estado Articulo 5", k=12)
    for d in docs:
        text = normalize_text(d.page_content)
        # Busca frase exacta, incluso si es corta (evita perderla por filtros de split).
        m = re.search(
            r"(La\s+capital\s+del\s+Estado\s+es\s+Madrid\.?)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            d.metadata["source"] = d.metadata.get("source", "desconocido")
            return m.group(1).strip(), [d]

        # Fallback: frase que contenga "capital del Estado".
        for fragment in re.split(r"(?<=[\.\!\?])\s+", text):
            frag = fragment.strip()
            if not frag:
                continue
            if "capital del estado" in strip_accents(frag.lower()):
                d.metadata["source"] = d.metadata.get("source", "desconocido")
                return frag, [d]
    return "", []


def answer_rights_listing_query(vectorstore: FAISS, question: str):
    q = strip_accents(question.lower())
    if not any(
        k in q
        for k in [
            "que derechos hay",
            "que derechos aparecen",
            "todos los derechos",
            "enumera los derechos",
            "lista los derechos",
        ]
    ):
        return "", []

    docs = vectorstore.similarity_search(
        "Se reconocen y protegen los derechos articulo 20 articulo 14 articulo 15",
        k=20,
    )
    rights = []
    used_docs = []
    seen_rights = set()

    for d in docs:
        text = normalize_text(d.page_content)
        # Captura items tipo "a) ...", "b) ...", etc. frecuentes en Art. 20.
        for m in re.finditer(r"(?:^|\s)([a-f]\)\s+[^;:.]{20,220})", text, flags=re.IGNORECASE):
            item = m.group(1).strip()
            key = strip_accents(item.lower())
            if key in seen_rights:
                continue
            seen_rights.add(key)
            rights.append(item)
            if d not in used_docs:
                used_docs.append(d)
            if len(rights) >= 6:
                break

        if len(rights) >= 6:
            break

    # Fallback por articulos conocidos de derechos fundamentales.
    if not rights:
        fallback_docs = vectorstore.similarity_search("articulo 14 igualdad articulo 15 vida integridad articulo 24 tutela judicial", k=8)
        snippets = []
        for d in fallback_docs:
            text = normalize_text(d.page_content)
            for frag in re.split(r"(?<=[\.\!\?])\s+", text):
                f = frag.strip()
                if len(f) < 40:
                    continue
                f_norm = strip_accents(f.lower())
                if any(k in f_norm for k in ["igualdad", "vida", "integridad", "tutela efectiva", "libertad"]):
                    snippets.append(f)
                    if d not in used_docs:
                        used_docs.append(d)
                    if len(snippets) >= 4:
                        break
            if len(snippets) >= 4:
                break
        if snippets:
            answer = "Segun los documentos, algunos derechos destacados son: " + " | ".join(
                [f"{i + 1}) {s}" for i, s in enumerate(snippets)]
            )
            return answer, used_docs[:2]
        return "", []

    answer = "Segun los documentos, se reconocen entre otros estos derechos: " + " | ".join(
        [f"{i + 1}) {r}" for i, r in enumerate(rights)]
    )
    return answer, used_docs[:2]


def answer_legal_consequence_query(vectorstore: FAISS, question: str):
    q = strip_accents(question.lower())
    if "vulner" not in q or "derecho" not in q:
        return "", []

    docs = vectorstore.similarity_search(
        "recurso de amparo vulneracion derechos fundamentales tribunal constitucional",
        k=12,
    )
    for d in docs:
        text = normalize_text(d.page_content)
        for sent in split_sentences(text):
            sent_norm = strip_accents(sent.lower())
            if "recurso de amparo" in sent_norm or "articulo 53" in sent_norm:
                d.metadata["source"] = d.metadata.get("source", "desconocido")
                return (
                    f"Segun los documentos, ante la vulneracion de derechos fundamentales procede el recurso de amparo: {sent}",
                    [d],
                )
            if "tribunal constitucional" in sent_norm and any(
                k in sent_norm for k in ["norma", "ley", "contraria", "inconstitucional"]
            ):
                d.metadata["source"] = d.metadata.get("source", "desconocido")
                return (
                    f"Segun los documentos, una via de proteccion ante la vulneracion es: {sent}",
                    [d],
                )
    return "", []


def get_all_index_docs(vectorstore: FAISS):
    # LangChain FAISS guarda documentos en docstore interno.
    store = getattr(vectorstore, "docstore", None)
    data = getattr(store, "_dict", {}) if store is not None else {}
    return list(data.values()) if isinstance(data, dict) else []


def is_heading_like_article(text: str, article_num: int) -> bool:
    # Verifica que el bloque comience efectivamente por "Articulo/Artículo N".
    prefix = text[:80]
    pattern = rf"^\s*(?:Artículo|Articulo)\s+{article_num}(?:\.|º|ª|\.º|\.ª)?\s+"
    return bool(re.search(pattern, prefix, flags=re.IGNORECASE))


def compose_answer_by_intent(candidates: List[Dict[str, object]], question: str, intent: str):
    if intent == "definition":
        target = canonical_token(definition_target(question))
        for c in candidates:
            sent = str(c["sentence"])
            sent_norm = strip_accents(sent.lower())
            sent_can = canonical_tokenized(sent_norm)
            if target and target in sent_can and " es " in sent_norm:
                return sent, [str(c["source"])]
        return "", []

    if intent == "comparative":
        top = candidates[:15]
        first = top[0] if top else None
        if first and int(first["score"]) < 4:
            return "", []

        q_norm = strip_accents(question.lower())
        # Regla especifica robusta para comparacion constitucional comun.
        if "derechos fundamentales" in q_norm and "principios rectores" in q_norm:
            derechos = find_sentence_with_phrase(
                top, ["derechos fundamentales", "seccion 1", "articulo 15", "articulo 14"]
            )
            principios = find_sentence_with_phrase(
                top, ["principios rectores", "capitulo iii", "politica social y economica"]
            )
            if derechos and principios:
                text = (
                    "Segun los documentos, la diferencia principal es: "
                    f"los derechos fundamentales se regulan en el Titulo I (especialmente su Seccion 1), "
                    f"mientras que los principios rectores orientan la politica social y economica. "
                    f"Evidencia 1: {derechos['sentence']} Evidencia 2: {principios['sentence']}"
                )
                return text, [str(derechos["source"]), str(principios["source"])]

        topic_a, topic_b = extract_comparison_topics(question)
        if topic_a and topic_b:
            a_hit = pick_sentence_with_topic(top, topic_a)
            b_hit = pick_sentence_with_topic(
                top,
                topic_b,
                forbidden_sentence=str(a_hit["sentence"]) if a_hit else "",
            )
            if a_hit and b_hit:
                text = (
                    "Segun los documentos, la comparacion principal es: "
                    f"{a_hit['sentence']} En contraste, {b_hit['sentence']}"
                )
                return text, [str(a_hit["source"]), str(b_hit["source"])]

        second = None
        for c in top[1:]:
            if c["source"] != first["source"]:
                second = c
                break
        if first and second:
            text = (
                "Segun los documentos, la comparacion principal es: "
                f"{first['sentence']} En contraste, {second['sentence']}"
            )
            return text, [str(first["source"]), str(second["source"])]
        if first:
            return str(first["sentence"]), [str(first["source"])]
        return "", []

    if intent == "consequence":
        for c in candidates:
            sent_norm = strip_accents(str(c["sentence"]).lower())
            if any(m in sent_norm for m in ["consecuencia", "provoca", "causa", "genera", "implica", "por lo tanto"]):
                return (
                    f"Segun los documentos, la consecuencia indicada es: {c['sentence']}",
                    [str(c["source"])],
                )
        for c in candidates:
            sent_norm = strip_accents(str(c["sentence"]).lower())
            if any(m in sent_norm for m in ["amparo", "tribunal constitucional", "nulo", "podra recurrir", "inconstitucional"]):
                return (
                    f"Segun los documentos, una consecuencia o via de actuacion es: {c['sentence']}",
                    [str(c["source"])],
                )
        return "", []

    if intent == "integration":
        top = candidates[:15]
        by_source: Dict[str, Dict[str, object]] = {}
        for c in top:
            source = str(c["source"])
            if source not in by_source:
                by_source[source] = c
        selected = list(by_source.values())[:2]
        if len(selected) < 2:
            return "", []
        text = (
            "Integrando informacion de dos documentos: "
            f"{selected[0]['sentence']} Ademas, {selected[1]['sentence']}"
        )
        return text, [str(selected[0]["source"]), str(selected[1]["source"])]

    if intent == "summary":
        top = candidates[:12]
        q_norm = strip_accents(question.lower())
        if "modelo de estado" in q_norm:
            line1 = pick_sentence_with_regex(
                top, r"estado social y democratico de derecho|monarquia parlamentaria"
            )
            line2 = pick_sentence_with_regex(
                top, r"soberania nacional reside en el pueblo|soberania nacional"
            )
            line3 = pick_sentence_with_regex(
                top, r"unidad de la nacion espanola|autonomia de las nacionalidades y regiones"
            )
            if line1 and line2 and line3:
                summary = (
                    "Ideas clave segun los documentos: "
                    f"1) {line1['sentence']} "
                    f"2) {line2['sentence']} "
                    f"3) {line3['sentence']}"
                )
                return summary, [str(line1["source"]), str(line2["source"]), str(line3["source"])]

        picks = []
        used_sources = []
        for c in top:
            sentence = str(c["sentence"])
            if len(sentence) < 60:
                continue
            picks.append(sentence)
            src = str(c["source"])
            if src not in used_sources:
                used_sources.append(src)
            if len(picks) >= 3:
                break
        if not picks:
            return "", []
        if len(picks) == 1:
            return picks[0], used_sources
        summary = "Ideas clave segun los documentos: " + " ".join(
            [f"{idx + 1}) {txt}" for idx, txt in enumerate(picks)]
        )
        return summary, used_sources

    # factual/general
    best = candidates[0]
    return str(best["sentence"]), [str(best["source"])]


def extract_comparison_topics(question: str):
    q = strip_accents(question.lower())
    q = re.sub(r"[¿?]", "", q)
    m = re.search(r"entre\s+(.+?)\s+y\s+(.+?)(?:\s+segun|\s*$)", q)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m2 = re.search(r"compara\s+(.+?)\s+y\s+(.+?)(?:\s+segun|\s*$)", q)
    if m2:
        return m2.group(1).strip(), m2.group(2).strip()
    return "", ""


def pick_sentence_with_topic(
    candidates: List[Dict[str, object]], topic: str, forbidden_sentence: str = ""
):
    topic_terms = [canonical_token(t) for t in re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ]+", topic) if len(t) > 2]
    if not topic_terms:
        return None
    for c in candidates:
        if forbidden_sentence and str(c["sentence"]).strip() == forbidden_sentence.strip():
            continue
        sent_norm = strip_accents(str(c["sentence"]).lower())
        sent_can = canonical_tokenized(sent_norm)
        if any(t in sent_can for t in topic_terms):
            return c
    return None


def find_sentence_with_phrase(candidates: List[Dict[str, object]], phrases: List[str]):
    for c in candidates:
        sent_norm = strip_accents(str(c["sentence"]).lower())
        if any(p in sent_norm for p in phrases):
            return c
    return None


def pick_sentence_with_regex(candidates: List[Dict[str, object]], pattern: str):
    rx = re.compile(pattern, flags=re.IGNORECASE)
    for c in candidates:
        sent_norm = strip_accents(str(c["sentence"]).lower())
        if rx.search(sent_norm):
            return c
    return None


def definition_target(question: str) -> str:
    q = strip_accents(question.lower())
    q = re.sub(r"[¿?]", "", q)
    patterns = [
        r"que es (.+)",
        r"que significa (.+)",
        r"definicion de (.+)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if m:
            raw = m.group(1).strip()
            raw = re.sub(r"\?$", "", raw).strip()
            raw = re.sub(r"^(la|el|los|las)\s+", "", raw)
            return raw
    return ""


def find_direct_definition(vectorstore: FAISS, question: str) -> str:
    target = definition_target(question)
    if not target:
        return ""

    # Busca evidencia especificamente definitoria para el termino.
    probe_query = f"{target} es"
    probe_docs = vectorstore.similarity_search(probe_query, k=8)
    target_canonical = canonical_token(target)

    candidates: List[str] = []
    for d in probe_docs:
        text = normalize_text(d.page_content)
        for sent in split_sentences(text):
            if is_bad_sentence(sent):
                continue
            sent_norm = strip_accents(sent.lower())
            sent_can = canonical_tokenized(sent_norm)
            has_term = target_canonical in sent_can
            has_is_pattern = " es " in sent_norm
            if has_term and has_is_pattern:
                candidates.append(sent.strip())

    # Devuelve la primera definicion limpia y suficientemente informativa.
    for c in candidates:
        if 50 <= len(c) <= 260:
            return c
    return candidates[0] if candidates else ""


def select_docs_by_sources(docs, sources: List[str]):
    selected = []
    seen = set()
    for source in sources:
        for d in docs:
            d_source = d.metadata.get("source", "desconocido")
            if d_source == source and source not in seen:
                selected.append(d)
                seen.add(source)
                break
    return selected


def format_sources(source_docs) -> List[str]:
    seen = set()
    ordered_sources = []
    for doc in source_docs:
        source = doc.metadata.get("source", "desconocido")
        if source not in seen:
            seen.add(source)
            ordered_sources.append(source)
    return ordered_sources


def interactive_chat(vectorstore: FAISS):
    print("\nChatbot documental listo.")
    print("Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        question = input("Tu pregunta: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            print("Sesion finalizada.")
            break

        if not question:
            print("Ingresa una pregunta valida.\n")
            continue

        answer, source_docs = answer_with_retrieval(vectorstore=vectorstore, question=question)
        sources = format_sources(source_docs)

        print("\nRespuesta:")
        print(answer)
        print("\nDocumento(s) fuente:")
        if sources:
            for s in sources:
                print(f"- {s}")
        else:
            print("- No identificado")
        print("")


def run_prompt_batch(vectorstore: FAISS, prompts_file: Path, output_file: Path):
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"No existe el archivo de prompts: {prompts_file}. "
            "Crea un archivo con una pregunta por linea."
        )

    raw_lines = prompts_file.read_text(encoding="utf-8").splitlines()
    prompts = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise ValueError("El archivo de prompts no contiene preguntas validas.")

    rows = []
    for idx, question in enumerate(prompts, start=1):
        answer, source_docs = answer_with_retrieval(vectorstore=vectorstore, question=question)
        sources = format_sources(source_docs)
        source_text = ", ".join(sources) if sources else "No identificado"
        rows.append((idx, question, answer, source_text))

    lines = [
        "# Resultados de evaluacion de prompts",
        "",
        "| # | Prompt | Respuesta del chatbot | Documento(s) fuente |",
        "|---|---|---|---|",
    ]
    for idx, prompt, answer, source_text in rows:
        safe_prompt = prompt.replace("|", "\\|")
        safe_answer = answer.replace("|", "\\|")
        safe_source = source_text.replace("|", "\\|")
        lines.append(f"| {idx} | {safe_prompt} | {safe_answer} | {safe_source} |")

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluacion completada. Archivo generado: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chatbot RAG para consulta sobre documentos personalizados."
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="data/docs",
        help="Carpeta con documentos PDF/TXT/MD.",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/faiss_index",
        help="Carpeta para guardar/cargar el indice vectorial.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Regenera el indice vectorial desde cero.",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default="",
        help="Archivo con preguntas (una por linea) para evaluacion automatica.",
    )
    parser.add_argument(
        "--prompts-output",
        type=str,
        default="resultados_prompts.md",
        help="Salida en Markdown de la evaluacion por prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    docs_dir = Path(args.docs)
    index_dir = Path(args.index)

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(index_dir.parent, exist_ok=True)
    if args.rebuild_index and index_dir.exists():
        shutil.rmtree(index_dir)

    vectorstore = build_or_load_vectorstore(docs_dir=docs_dir, index_dir=index_dir)
    if args.prompts_file:
        run_prompt_batch(
            vectorstore=vectorstore,
            prompts_file=Path(args.prompts_file),
            output_file=Path(args.prompts_output),
        )
    else:
        interactive_chat(vectorstore)


if __name__ == "__main__":
    main()
