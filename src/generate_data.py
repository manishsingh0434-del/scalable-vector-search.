import argparse
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Each topic has complete, technically coherent passages.
PASSAGES = {
    "Python": [
        "Python generators use yield to produce values incrementally. This can reduce peak memory usage when processing large sequences.",
        "A virtual environment isolates Python packages for a project. This helps prevent dependency conflicts between applications.",
        "Python exceptions should be handled using specific exception types. Logging the exception with relevant context makes failures easier to diagnose.",
        "Unit tests validate individual functions, while integration tests check interactions between components. Both can help detect regressions.",
        "Pinned dependency versions make Python installations more reproducible. Test package upgrades before deploying them.",
        "Context managers release resources reliably when a block exits. The with statement is commonly used for files and locks.",
        "Python logging supports severity levels such as DEBUG, INFO, WARNING, and ERROR. Appropriate levels help operators investigate failures.",
        "Type hints document expected values and improve static analysis. They do not automatically validate every value at runtime.",
        "Streaming a large file line by line avoids loading the entire file into memory. This is useful for large logs and data exports.",
        "Profiling measures where a program spends time or memory. Measurements help identify bottlenecks before optimization.",
    ],
    "Docker": [
        "Docker images are built from Dockerfiles. Multi-stage builds separate build tools from runtime dependencies and can reduce final image size.",
        "Docker volumes store data outside a container's writable layer. They are useful when data must survive container replacement.",
        "Container health checks can detect an application that is running but not responding correctly. Health status should reflect useful application behavior.",
        "Container logs should be sent to a centralized logging system. This makes failures easier to investigate across multiple services.",
        "Docker image layers can be reused between builds. Ordering instructions carefully can improve build-cache efficiency.",
        "Running containers as non-root users reduces the impact of some application compromises. Images should grant only necessary permissions.",
        "Container CPU and memory limits help prevent a single workload from exhausting host resources. Limits should be based on observed workload needs.",
        "Image tags identify versions, but mutable tags can point to different images over time. Digests provide immutable image references.",
        "Docker bridge networks allow containers to communicate on a shared network. Published ports expose selected container services to the host.",
        "A container's environment variables can configure application behavior. Secrets should not be baked into image layers.",
    ],
    "Kubernetes": [
        "Readiness probes determine whether a Pod should receive traffic. Liveness probes help restart containers that cannot recover.",
        "Startup probes give slow-starting applications time to initialize before liveness checks begin. They help avoid premature restarts.",
        "Kubernetes Deployments maintain a desired number of application replicas and support controlled rolling updates.",
        "Kubernetes Services provide stable network access to changing groups of Pods. Selectors connect Services to matching workloads.",
        "Horizontal Pod Autoscaling adjusts replica counts using configured metrics such as CPU utilization. It requires suitable metrics to be available.",
        "Resource requests help the Kubernetes scheduler place Pods on nodes. Resource limits constrain CPU or memory consumption.",
        "ConfigMaps store non-sensitive configuration for Kubernetes workloads. Secrets are intended for sensitive values and require appropriate access controls.",
        "A Pod disruption budget limits voluntary disruptions to an application. It does not prevent every possible outage.",
        "NetworkPolicies can restrict traffic between Pods when the cluster network plugin supports their enforcement.",
        "PersistentVolumeClaims request storage for workloads. The available storage class and access mode affect how volumes can be provisioned.",
        "Rolling updates gradually replace application replicas. Readiness checks help prevent unavailable replicas from receiving traffic.",
        "Kubernetes scheduling constraints influence which nodes can run a Pod. Resource availability and node labels can affect placement.",
    ],
    "Vector Search": [
        "An embedding model converts text into a numerical vector. Semantically related text can have vectors that are close under a suitable distance metric.",
        "Cosine distance compares vector direction. With common normalized embedding models, lower cosine distance generally indicates greater similarity.",
        "Document chunking divides long text into smaller passages. Poor chunk boundaries can separate a fact from the context needed to understand it.",
        "Recall at k measures whether known relevant documents appear among the first k results. It requires a defined set of relevant documents.",
        "Duplicate documents can crowd nearest-neighbor results. Deduplication can improve result diversity and make retrieval evaluation more informative.",
        "A vector index trades memory use, build time, search latency, and retrieval quality. Representative workloads are needed to evaluate these trade-offs.",
        "Metadata filtering restricts vector search to records that match specified attributes. Filters can help narrow results to a source or category.",
        "Embedding dimensions affect storage and computation requirements. Models should be evaluated for quality as well as vector size.",
        "Batching embedding requests can reduce repeated overhead. Batch size must fit available memory and the model's supported input limits.",
        "A persistent vector database saves indexed records between program runs. Reopening the same storage path allows an application to reuse the collection.",
        "Semantic search retrieves content by vector similarity rather than requiring exact keyword matches. It can still miss relevant facts when embeddings rank them poorly.",
        "Retrieval evaluation should use representative queries and judged relevant passages. Topic matching alone does not prove that a result answers the question.",
    ],
    "Batch Processing": [
        "Batch ingestion processes records in bounded groups to control peak memory usage. Throughput should be measured alongside resource consumption.",
        "A checkpoint records completed work so an interrupted batch job can resume without repeating every earlier step.",
        "Idempotent writes make retries safer by preventing repeated execution from creating unintended duplicate records.",
        "Backpressure slows producers when downstream workers cannot keep up. It helps prevent unbounded queues and excessive memory use.",
        "Batch throughput measures records processed per unit of time. Record processing time, failure rate, and total job duration as well.",
        "Retry policies should distinguish temporary failures from permanent input errors. Unbounded retries can waste resources and delay recovery.",
        "A dead-letter queue can hold records that repeatedly fail processing. Operators can inspect them without blocking all other work.",
        "Input validation catches malformed records before expensive processing begins. Validation errors should include enough context to identify the cause.",
        "Worker concurrency should be matched to CPU, memory, and downstream service capacity. More workers do not always increase throughput.",
        "Progress counters make long-running batch jobs easier to monitor. Track completed, failed, and remaining records.",
    ],
    "Observability": [
        "Application latency measures how long requests take. Percentiles such as p95 help reveal slow requests that an average can hide.",
        "Error rate measures the proportion of operations that fail. A sudden increase can indicate regressions or dependency outages.",
        "Logs provide event details, metrics summarize numerical behavior, and traces connect operations across distributed services.",
        "CPU utilization, memory pressure, disk capacity, and request latency help identify resource bottlenecks before they cause outages.",
        "An actionable alert identifies a condition that requires investigation. Alerts based on sustained symptoms can reduce noise.",
        "Request throughput measures how many operations a service completes over time. Interpret it alongside latency and errors.",
        "A service-level objective defines a target for a reliability measure over a specified period. Error budgets help teams reason about risk.",
        "Distributed trace spans represent individual operations within a request. Trace context connects spans across services.",
        "Structured logs store fields in a consistent format. This makes filtering and aggregation easier than parsing arbitrary text.",
        "Monitoring dashboards should present metrics that support a specific operational question. Too many unrelated charts can obscure important signals.",
    ],
    "Linux": [
        "The top and ps commands inspect running processes. They help identify processes consuming excessive CPU or memory.",
        "The df -h command reports filesystem capacity in readable units. The du -sh command summarizes the disk usage of a directory.",
        "The free -h command summarizes memory and swap usage. Available memory is often more useful than free memory alone.",
        "The journalctl command reads systemd journal logs. Filtering by service and time can help diagnose startup failures.",
        "The chmod command changes file permissions, while chown changes ownership. Incorrect permissions can prevent applications from reading files.",
        "The ss command displays socket information. It can help investigate listening ports and network connections.",
        "Systemd services can be inspected with systemctl status. Service logs often provide more detail about startup failures.",
        "Linux load averages describe runnable or uninterruptible tasks over time. Interpret them alongside CPU count and resource utilization.",
        "Filesystem inode exhaustion can prevent file creation even when disk space remains. The df -i command reports inode usage.",
        "SSH authentication failures can be investigated using system authentication logs. Key permissions and server configuration are common areas to check.",
        "The kill command sends signals to processes. Prefer graceful termination before using a signal that forces a process to stop.",
        "Shell pipelines connect the output of one command to another. Small focused commands can make system diagnostics easier to reproduce.",
    ],
    "Databases": [
        "Database indexes can speed up selective queries but consume storage and add overhead to inserts and updates.",
        "A query execution plan shows how a database intends to execute a query. It can reveal full table scans and inefficient joins.",
        "Connection pooling reuses database connections and reduces setup overhead. Pool size must respect database capacity.",
        "Transactions group related operations into a unit of work. Isolation settings influence how concurrent transactions interact.",
        "Database backups should be tested through restoration exercises. A successful backup job alone does not guarantee recoverability.",
        "Composite indexes cover multiple columns. Column order and the query's filtering pattern influence whether the index is useful.",
        "Slow-query logs help identify expensive database operations. Compare query plans and execution times before changing indexes.",
        "Database locks coordinate concurrent access to data. Long transactions can increase contention and delay other operations.",
        "Connection limits protect a database from excessive concurrent sessions. Applications should reuse connections and handle pool exhaustion.",
        "Keyset pagination uses values from the previous page to find the next set of rows. It can be more stable than large offset scans.",
        "Write-ahead logging records changes before data pages are written. It supports recovery and, in some systems, replication.",
        "Schema migrations should be planned for compatibility between application versions. Backups and rollback plans reduce deployment risk.",
    ],
}


def generate_documents(count, seed):
    """Create unique documents by combining each topic's passage library."""
    rng = random.Random(seed)
    topic_names = list(PASSAGES)
    documents = []

    # Shuffle the passage order within each topic reproducibly.
    pools = {}
    for topic, passages in PASSAGES.items():
        pool = list(passages)
        rng.shuffle(pool)
        pools[topic] = pool

    for index in range(count):
        topic = topic_names[index % len(topic_names)]
        passages = pools[topic]
        cycle = index // len(topic_names)

        if cycle < len(passages):
            passage = passages[cycle]
        else:
            # After the unique library is exhausted, repeat explicitly
            # with a clear identifier for load testing.
            base = passages[cycle % len(passages)]
            passage = (
                f"Load-test record {index + 1}. {base} "
                f"Synthetic record identifier: {index + 1}."
            )

        documents.append((topic, passage))

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Generate a technical corpus for vector-search testing."
    )
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing synthetic documents only.",
    )
    args = parser.parse_args()

    if args.documents < 1:
        parser.error("--documents must be at least 1")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.replace:
        for path in DATA_DIR.glob("synthetic_*.txt"):
            path.unlink()

    documents = generate_documents(args.documents, args.seed)
    seen = set()

    for index, (topic, passage) in enumerate(documents):
        content = f"Topic: {topic}\n\n{passage}\n"
        normalized = " ".join(content.split())

        if normalized in seen:
            raise RuntimeError(
                f"Duplicate content detected at generated document {index}"
            )
        seen.add(normalized)

        output = DATA_DIR / f"synthetic_{index:06d}.txt"
        output.write_text(content, encoding="utf-8")

        if (index + 1) % 1000 == 0:
            print(f"Generated {index + 1} documents")

    print("\nGeneration complete.")
    print(f"Synthetic documents generated: {len(documents)}")
    print(f"Unique generated documents: {len(seen)}")
    print(f"Output directory: {DATA_DIR}")
    print(
        "Note: records beyond the unique passage library are explicitly "
        "labelled load-test records, not new independent knowledge."
    )


if __name__ == "__main__":
    main()
