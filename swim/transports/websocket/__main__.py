import asyncio

from swim.transports.websocket.server import build_parser, serve


def main():
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
