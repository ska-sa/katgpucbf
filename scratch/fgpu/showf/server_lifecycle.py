import sys
import asyncio

from backend import Backend, parse_args


_argv = sys.argv


def on_server_loaded(server_context):
    args = parse_args(_argv[1:])
    backend = Backend('239.102.200.0:7148', 'p4p1', args.channels, args.channels, args.acc_len, args.keep_ratio, server_context)
    print(type(server_context))
    asyncio.get_event_loop().create_task(backend.run())
    print('Server loaded')
