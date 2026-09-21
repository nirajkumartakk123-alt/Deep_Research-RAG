import pytest

from app.core.observability import track_node, track_node_sync


@pytest.mark.asyncio
async def test_track_node_adds_trace_entry():
    @track_node("dummy_node")
    async def dummy(state: dict) -> dict:
        return {"foo": "bar"}

    result = await dummy({})
    assert result["foo"] == "bar"
    assert "run_metadata" in result
    assert result["run_metadata"]["trace"][0]["node"] == "dummy_node"
    assert result["run_metadata"]["trace"][0]["status"] == "success"


@pytest.mark.asyncio
async def test_track_node_accumulates_multiple_traces():
    @track_node("node_a")
    async def node_a(state: dict) -> dict:
        return {}

    @track_node("node_b")
    async def node_b(state: dict) -> dict:
        return {}

    state = {}
    state.update(await node_a(state))
    state.update(await node_b(state))

    trace_nodes = [t["node"] for t in state["run_metadata"]["trace"]]
    assert trace_nodes == ["node_a", "node_b"]


def test_track_node_sync_adds_trace_entry():
    @track_node_sync("dummy_sync_node")
    def dummy(state: dict) -> dict:
        return {"foo": "bar"}

    result = dummy({})
    assert result["run_metadata"]["trace"][0]["node"] == "dummy_sync_node"