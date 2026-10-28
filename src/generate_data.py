
import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TOPICS = {
    "Python": [
        "Python virtual environments isolate project dependencies. Use python -m venv .venv to create one and activate it before installing packages.",
        "Python exceptions should be handled with specific exception types. Logging errors with context makes debugging production failures easier.",
        "Automated tests validate expected behavior. Unit tests focus on small functions, while integration tests verify interactions between components.",
        "Python generators yield values incrementally instead of building an entire result in memory. They are useful when processing large datasets.",
        "Dependency files make Python environments reproducible. Pin compatible package versions and test upgrades before deployment.",
    ],
    "Docker": [
        "Docker images are built from Dockerfiles. Multi-stage builds can reduce final image size by separating build dependencies from runtime dependencies.",
        "Container logs should be sent to a centralized logging system so operators can diagnose failures across multiple services.",
        "Docker volumes persist data beyond the lifecycle of an individual container. Use them when application state must survive container replacement.",
        "Container health checks help detect unhealthy applications. A running container does not necessarily mean its application is ready.",
        "Container resource limits can prevent one workload from consuming all available CPU and memory on a host.",
    ],
    "Kubernetes": [
        "Kubernetes Deployments maintain a desired number of application replicas and support controlled rolling updates.",
        "Kubernetes Services provide stable network access to changing groups of Pods. Selectors connect Services to matching workloads.",
        "Readiness probes determine whether a Pod should receive traffic. Liveness probes help restart containers that cannot recover.",
        "Horizontal Pod Autoscaling adjusts replica counts using metrics such as CPU utilization when resource metrics are configured.",
        "Kubernetes resource requests help the scheduler place Pods, while resource limits constrain their CPU and memory consumption.",
    ],
    "Vector Search": [
        "An embedding model converts text into a numerical vector. Similar meanings can produce vectors that are close under a chosen distance metric.",
        "Cosine distance compares vector direction and is commonly used for semantic text retrieval. Lower cosine distance generally means greater similarity.",
        "Chunking splits long documents into smaller passages. Very large chunks can introduce irrelevant context, while tiny chunks may lose meaning.",
        "Recall at k measures whether relevant documents appear among the first k retrieved results. It evaluates retrieval separately from answer generation.",
        "A vector index trades memory, indexing time, search latency, and recall. Evaluate these trade-offs using representative queries.",
        "Duplicate documents can dominate nearest-neighbor results. Deduplication and diverse evaluation data make search quality measurements more useful.",
    ],
    "Batch Processing": [
        "Batch ingestion processes records in bounded groups to control peak memory usage. Batch size should be tuned using throughput and resource measurements.",
        "Retries can recover from temporary failures, but idempotent writes are important to avoid duplicating records during repeated ingestion.",
        "A checkpoint records completed work so a failed batch job can resume without repeating every earlier operation.",
        "Batch throughput is the number of records processed per unit of time. Measure it alongside failure rates and total job duration.",
        "Backpressure slows producers when downstream components cannot keep up, preventing unbounded queues and excessive memory consumption.",
    ],
    "Observability": [
        "Application latency measures how long requests take. Track median and tail percentiles such as p95 to understand slow requests.",
        "Error rate is the proportion of operations that fail. Sudden increases can reveal regressions, dependency outages, or configuration problems.",
        "Logs provide event details, metrics summarize numerical behavior, and traces connect operations across distributed services.",
        "Memory utilization, disk capacity, CPU usage, and request latency help identify resource bottlenecks before they cause outages.",
        "A useful alert should identify an actionable problem. Alerts based on sustained symptoms are often more useful than noisy single measurements.",
    ],
    "Linux": [
        "The top and ps commands inspect running processes. They help identify processes consuming excessive CPU or memory.",
        "The df -h command reports filesystem capacity in readable units. The du -sh command summarizes the disk usage of a directory.",
        "The free -h command summarizes memory and swap usage. Available memory is often more useful than free memory alone on Linux.",
        "The journalctl command reads systemd journal logs. Filtering by service and time can help diagnose startup failures.",
        "Linux file permissions control read, write, and execute access. The chmod and chown commands modify permissions and ownership.",
        "The ss command displays socket information. It can help investigate listening ports and network connections.",
    ],
    "Databases": [
        "Database indexes can speed up selective queries but consume storage and add overhead to inserts and updates.",
        "A query execution plan shows how a database intends to execute a query. It can reveal full table scans and inefficient joins.",
        "Connection pooling reuses database connections and reduces setup overhead, but pool size must match database capacity.",
        "Transactions group related operations into a unit of work. Atomicity and isolation help preserve consistency under concurrent access.",
        "Database backups should be tested through restoration exercises. A successful backup job alone does not guarantee recoverability.",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate a clean local technical corpus."
    )
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing synthetic documents.",
    )
    args = parser.parse_args()

    if args.documents < 1:
        parser.error("--documents must be at least 1")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    topic_names = list(TOPICS)

    if args.replace:
        for path in DATA_DIR.glob("synthetic_*.txt"):
            path.unlink()

    for index in range(args.documents):
        topic = topic_names[index % len(topic_names)]
        passages = TOPICS[topic]
        passage = passages[(index // len(topic_names)) % len(passages)]

        # Keep one technical passage per document.
        # Vary only whitespace-free context in the query-independent metadata.
        content = f"Topic: {topic}\n\n{passage}\n"

        output = DATA_DIR / f"synthetic_{index:06d}.txt"
        output.write_text(content, encoding="utf-8")

        if (index + 1) % 1000 == 0:
            print(f"Generated {index + 1} documents")

    print("\nGeneration complete.")
    print(f"Synthetic documents generated: {args.documents}")
    print(f"Output directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
