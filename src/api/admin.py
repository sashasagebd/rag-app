"""Admin tasks for the web app, from the command line (on the server).

Usage (from the repo root):
    python -m api.admin invite [--note "for Sam"]      # prints a one-time invite code
    python -m api.admin users                           # accounts, keys, usage
    python -m api.admin server-key sasha on|off         # allow/deny the server's API key
    python -m api.admin limit sasha 100|default         # questions per day for one user

Uses APP_CONFIG (default configs/app.toml) for the database URL.
"""

import argparse
import os
import sys

from api import accounts
from api.accounts import AccountError
from common.config import load_config
from ingest.embed_load import connect, database_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("invite")
    p.add_argument("--note", help="who it's for (stored with the invite)")
    sub.add_parser("users")
    p = sub.add_parser("server-key")
    p.add_argument("username")
    p.add_argument("state", choices=["on", "off"])
    p = sub.add_parser("limit")
    p.add_argument("username")
    p.add_argument("questions", help="questions per day, or 'default'")
    args = parser.parse_args()

    config = load_config(os.environ.get("APP_CONFIG", "configs/app.toml"))
    try:
        with connect(database_url(config)) as conn:
            if args.command == "invite":
                code = accounts.create_invite(conn, args.note)
                print(f"invite code: {code}\n(one-time use; it is not stored, so copy it now)")
            elif args.command == "users":
                rows = accounts.list_users(conn)
                print(f"{'user':20} {'joined':10}  key   server-key  limit  today  total $")
                for name, created, has_key, server, limit, today, usd in rows:
                    print(f"{name:20} {created:%Y-%m-%d}  {'yes' if has_key else 'no ':3}   {'on ' if server else 'off'}"
                          f"         {limit or '-':>5}  {today:5}  {float(usd):7.3f}")
                if not rows:
                    print("no users yet: python -m api.admin invite")
            elif args.command == "server-key":
                accounts.update_user(conn, args.username, use_server_key=args.state == "on")
                print(f"{args.username}: server key {args.state}")
            elif args.command == "limit":
                n = None if args.questions == "default" else int(args.questions)
                accounts.update_user(conn, args.username, daily_question_limit=n)
                print(f"{args.username}: {n or 'default'} questions per day")
    except AccountError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
