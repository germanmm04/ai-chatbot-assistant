from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from chatbot_rag import answer_with_retrieval, build_or_load_vectorstore, format_sources


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interfaz web para el chatbot RAG."
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
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host donde se levantara la app web.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Puerto para la app web.",
    )
    return parser.parse_args()


def create_app(docs_dir: str, index_dir: str):
    app = Flask(__name__)
    vectorstore = build_or_load_vectorstore(
        docs_dir=Path(docs_dir),
        index_dir=Path(index_dir),
    )

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "La pregunta no puede estar vacia."}), 400

        answer, source_docs = answer_with_retrieval(
            vectorstore=vectorstore,
            question=question,
        )
        sources = format_sources(source_docs)
        return jsonify(
            {
                "answer": answer,
                "sources": sources if sources else ["No identificado"],
            }
        )

    return app


def main():
    args = parse_args()
    app = create_app(docs_dir=args.docs, index_dir=args.index)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
