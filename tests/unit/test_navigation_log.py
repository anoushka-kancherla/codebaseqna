import json
from server.navigation_log import NavigationLog


def test_file_read_records_entry():
    log = NavigationLog()
    log.record_file_read("a.py", "line1\nline2\n")
    entry = log.entries[0]
    assert entry.event == "file_read"
    assert entry.lines == 3


def test_tree_read_records_entry():
    log = NavigationLog()
    log.record_tree_read()
    entry = log.entries[0]
    assert entry.event == "tree_read"
    assert entry.path is None


def test_multiple_reads_ordered_by_timestamp():
    log = NavigationLog()
    log.record_file_read("a.py", "x")
    log.record_file_read("b.py", "y")
    log.record_file_read("c.py", "z")
    timestamps = [e.timestamp for e in log.entries]
    assert timestamps == sorted(timestamps)
    assert len(log.entries) == 3


def test_to_dict_serialisable():
    log = NavigationLog()
    log.record_file_read("a.py", "x")
    log.record_tree_read()
    log.record_tool_call("git_log", {"n": 5})
    json.dumps(log.to_dict())


def test_fresh_instance_has_no_entries():
    log = NavigationLog()
    assert log.entries == []


def test_path_traversal_rejected(tmp_path, monkeypatch):
    import server.mcp_server as mcp_server
    monkeypatch.setattr(mcp_server, "ROOT", tmp_path)
    monkeypatch.setattr(mcp_server, "NAV_LOG", NavigationLog())

    result = json.loads(mcp_server.get_file_json("../../etc/passwd"))
    assert result["error"] == "PATH_TRAVERSAL"
    assert mcp_server.NAV_LOG.entries == []


def test_binary_file_rejected(tmp_path, monkeypatch):
    import server.mcp_server as mcp_server
    binary_file = tmp_path / "image.bin"
    binary_file.write_bytes(b"\x00\x01\xff\xfe\x00binary")
    monkeypatch.setattr(mcp_server, "ROOT", tmp_path)
    monkeypatch.setattr(mcp_server, "NAV_LOG", NavigationLog())

    result = json.loads(mcp_server.get_file_json("image.bin"))
    assert result["error"] == "BINARY_FILE"
    assert mcp_server.NAV_LOG.entries == []


def test_large_file_truncated(tmp_path, monkeypatch):
    import server.mcp_server as mcp_server
    big_file = tmp_path / "big.txt"
    big_file.write_text("x" * (mcp_server.MAX_FILE_BYTES + 1000))
    monkeypatch.setattr(mcp_server, "ROOT", tmp_path)
    monkeypatch.setattr(mcp_server, "NAV_LOG", NavigationLog())

    result = json.loads(mcp_server.get_file_json("big.txt"))
    assert result["content"].endswith("[TRUNCATED — file exceeds 500 KB]")
    assert len(mcp_server.NAV_LOG.entries) == 1
