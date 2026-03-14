#!/usr/bin/env python3
"""
noVNC convenience wrapper — start/stop/status a VNC session for browser_login.

Usage:
    python3 novnc.py start    # Start VNC, print noVNC URL(s)
    python3 novnc.py stop     # Stop VNC session
    python3 novnc.py status   # Check if VNC is running
    python3 novnc.py urls     # Print all access URLs

This wraps vnc_session.py so the LLM (or user) can manage VNC from
run_server_command without importing Python modules.
"""

import sys


def main():
    from vnc_session import get_vnc_session

    vnc = get_vnc_session()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"

    if cmd == "start":
        result = vnc.start()
        print(result)
        if not result.startswith("Error"):
            urls = vnc.get_all_urls()
            if len(urls) > 1:
                print("\nAll access URLs:")
                for u in urls:
                    print(f"  {u}")
    elif cmd == "stop":
        print(vnc.stop())
    elif cmd == "status":
        if vnc.is_active:
            print(f"VNC is running on display {vnc.display}")
            for u in vnc.get_all_urls():
                print(f"  {u}")
        else:
            print("VNC is not running.")
    elif cmd == "urls":
        if vnc.is_active:
            for u in vnc.get_all_urls():
                print(u)
        else:
            print("VNC is not running. Start with: python3 novnc.py start")
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 novnc.py [start|stop|status|urls]")
        sys.exit(1)


if __name__ == "__main__":
    main()
