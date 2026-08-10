# Architecture

## 1. Overview

The system is a LangGraph `StateGraph` of eight agents, each wrapping one or
more stateless tools. State flows through a single typed `PipelineState`
(Pydantic-backed `TypedDict`) using LangGraph's reducer mechanism for
append-only fields (`execution_trace`, `errors`, `warnings`, `token_usage`).

## 2. Component diagram

```mermaid
flowchart TB
    subgraph API["FastAPI Layer"]
        R1["/api/v1/books/recognize"]
        R2["/api/v1/books/recognize/stream (SSE)"]
        R3["/health /ready /version /metrics"]
        MW1[ErrorHandlingMiddleware]
        MW2[RateLimitMiddleware]
        MW3[CORS + GZip]
    end

    subgraph Graph["LangGraph Orchestration"]
        C[Container - DI]
        WF[StateGraph Workflow]
        RT[Router - conditional edges]
        CK[(Checkpointer - SQLite/Memory)]
    end

    subgraph Agents["Agents (BaseAgent)"]
        A1[BookDetectionAgent]
        A2[ImageQualityAgent]
        A3[DecisionAgent]
        A4[SuperResolutionAgent]
        A5[GeminiVisionAgent]
        A6[ValidationAgent]
        A7[RecommendationAgent]
        A8[FinalResponseAgent]
    end

    subgraph Tools["Tools (BaseTool, stateless)"]
        T1[YOLODetectionTool]
        T2[ImageQualityTool]
        T3[SuperResolutionTool]
        T4[GeminiVisionTool]
        T5[GoogleBooksTool]
        T6[RecommendationTool]
    end

    subgraph External["External Services"]
        E1[(YOLO weights)]
        E2[Google Gemini API]
        E3[Google Books API]
        E4[(Redis Cache)]
    end

    R1 --> MW3 --> MW2 --> MW1 --> WF
    R2 --> WF
    WF --> C --> Agents
    WF --> RT
    WF --> CK
    A1 --> T1 --> E1
    A2 --> T2
    A4 --> T3
    A5 --> T4 --> E2
    A6 --> T5 --> E3
    A7 --> T6 --> E3
    T5 --> E4
    T6 --> E4
    T4 --> E4
```

## 3. Sequence diagram — happy path

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Graph as LangGraph
    participant Detect as BookDetectionAgent
    participant Quality as ImageQualityAgent
    participant Decide as DecisionAgent
    participant SR as SuperResolutionAgent
    participant Vision as GeminiVisionAgent
    participant Valid as ValidationAgent
    participant Rec as RecommendationAgent
    participant Final as FinalResponseAgent

    Client->>API: POST /api/v1/books/recognize (image)
    API->>API: validate_upload() + sanitize_and_reencode()
    API->>Graph: run_pipeline(image_path, request_id)
    Graph->>Detect: execute(state)
    Detect-->>Graph: detections[], cropped_images[]
    Graph->>Quality: execute(state)
    Quality-->>Graph: quality_reports[]
    Graph->>Decide: execute(state)
    Decide-->>Graph: tool_decisions[]
    alt needs super-resolution
        Graph->>SR: execute(state)
        SR-->>Graph: enhanced_images[]
    end
    Graph->>Vision: execute(state)
    Vision-->>Graph: vision_results[], token_usage
    Graph->>Valid: execute(state)
    Valid-->>Graph: validated_books[] (hallucinations dropped)
    Graph->>Rec: execute(state)
    Rec-->>Graph: recommendations[]
    Graph->>Final: execute(state)
    Final-->>Graph: metrics
    Graph-->>API: BookRecognitionResponse
    API-->>Client: 200 JSON
```

## 4. State model

`app/models/state.py::PipelineState` is the single contract threaded through
every node. Append-only fields use `Annotated[list[T], operator.add]` so
LangGraph merges partial node returns instead of overwriting history.
Agents therefore return **deltas**, not the full state — see the docstring
in `app/agents/base_agent.py` for the reducer-safety contract.

## 5. Routing logic

Implemented in `app/graph/router.py`, unit-tested in isolation from the
graph:

- `route_after_detection`: skip straight to `final_response` if YOLO found
  nothing.
- `route_after_decision`: send to `super_resolution` only if at least one
  crop needs it; skip straight to `gemini_vision` otherwise; skip both if
  every crop was rejected on quality.
- `route_after_validation`: skip `recommendation` if nothing survived
  Google Books validation.

## 6. Failure isolation

`BaseAgent.execute()` never lets an agent's exception propagate into the
graph. It's caught, logged with full context, recorded as a failed
`ExecutionStep` and an `errors` entry, and the graph continues — so a
transient Gemini outage degrades gracefully to a partial response (with
`status: "failed"` or `"partial_success"` in the API response) rather than
a 500.

## 7. Extending the pipeline

To add a new agent/stage:

1. Add any new tool under `app/tools/`, inheriting `BaseTool`.
2. Add the agent under `app/agents/`, inheriting `BaseAgent`, returning a
   partial-state delta from `run()`.
3. Register the tool + agent in `app/graph/container.py`.
4. Add the node + edges in `app/graph/workflow.py`.
5. If it needs new routing, add a function to `app/graph/router.py` and a
   unit test in `tests/unit/test_router.py`.
