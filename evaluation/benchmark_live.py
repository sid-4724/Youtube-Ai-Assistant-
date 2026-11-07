"""Measure end-to-end processing and first-answer latency against the local API."""
import argparse
import json
import re
import time

import httpx

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="11-character YouTube video ID")
    parser.add_argument("--question", default="What are the main ideas explained in this video?")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if not VIDEO_ID.fullmatch(args.video_id):
        parser.error("video_id must be exactly 11 URL-safe characters")

    started = time.perf_counter()
    with httpx.Client(base_url=args.api.rstrip("/"), timeout=600) as client:
        response = client.post(
            "/api/videos/initialize", json={"video_id": args.video_id, "languages": ["en"]}
        )
        response.raise_for_status()
        status = response.json()
        while status["status"] not in {"ready", "error"}:
            time.sleep(1)
            response = client.get(f"/api/videos/{args.video_id}/status")
            response.raise_for_status()
            status = response.json()
        processed_at = time.perf_counter()
        if status["status"] != "ready":
            raise SystemExit(json.dumps({"processing_error": status.get("error"), "status": status}))

        session_response = client.post("/api/sessions", json={"video_id": args.video_id})
        session_response.raise_for_status()
        session = session_response.json()
        answer_response = client.post(
            f"/api/videos/{args.video_id}/ask",
            json={
                "video_id": args.video_id,
                "session_id": session["session_id"],
                "question": args.question,
            },
        )
        answer_response.raise_for_status()
        answer = answer_response.json()
        finished = time.perf_counter()

    print(
        json.dumps(
            {
                "video_id": args.video_id,
                "question": args.question,
                "chunk_count": status["chunk_count"],
                "processing_seconds": round(processed_at - started, 3),
                "time_to_first_answer_seconds": round(finished - started, 3),
                "answer_generation_after_processing_seconds": round(finished - processed_at, 3),
                "abstained": answer["abstained"],
                "source_timestamps": [source["timestamp"] for source in answer["sources"]],
                "model_and_machine": "Record YTA_OLLAMA_MODEL, CPU/RAM/GPU, and warm/cold run conditions alongside this output.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
