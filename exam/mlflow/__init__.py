import mlflow
from mlflow.entities import SpanType


def calculate_overhead(run, total_time):
    """
    Calculate overhead time correctly handling parallel tool execution.
    It merges overlapping time intervals to avoid double-counting.
    """
    client = mlflow.MlflowClient()
    run_info = run.info
    total_duration_ms = total_time * 1000

    # Recupera tutte le trace per questa run
    traces = client.search_traces(
        locations=[run_info.experiment_id],
        run_id=run_info.run_id,
    )

    tool_intervals = []
    if traces:
        for trace in traces:
            for span in trace.data.spans:
                if span.span_type == SpanType.TOOL:
                    tool_intervals.append((span.start_time_ns, span.end_time_ns))

    merged_duration_ns = 0
    if tool_intervals:
        # Ordina per tempo di inizio
        tool_intervals.sort(key=lambda x: x[0])

        merged = []
        if tool_intervals:
            curr_start, curr_end = tool_intervals[0]

            for next_start, next_end in tool_intervals[1:]:
                if next_start < curr_end:
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end

            merged.append((curr_start, curr_end))

        for start, end in merged:
            merged_duration_ns += (end - start)

    tool_duration_ms = merged_duration_ns / 1_000_000
    overhead_ms = total_duration_ms - tool_duration_ms

    print("TIME ANALYSIS (PARALLEL CORRECTED)")
    print(f"Total Duration:       {total_duration_ms / 1000:.2f}s")
    print(f"Active Tool Time:     {tool_duration_ms / 1000:.2f}s")
    print(f"Overhead Time:        {overhead_ms / 1000:.2f}s")

    if total_duration_ms > 0:
        print(f"Overhead Percentage:  {(overhead_ms / total_duration_ms * 100):.1f}%")
    else:
        print("Overhead Percentage:  N/A")
