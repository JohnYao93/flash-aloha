"""Real loopback transport, synthetic Flow policy; no cameras or robot."""
import asyncio

import numpy as np
import torch
import websockets.asyncio.server

from inference.inference_remote import validate_policy_metadata
from inference.policy_transport import WebsocketOpenPIClient
from inference.serve_policy import LocalPolicyServer
from test_policy_server import DummyFlowPolicy, make_config, make_stats


def test_server_client_roundtrip_with_rtc_prefix():
    config = make_config()
    config['policy_config'].update(num_queries=50, training_time_rtc=True,
        training_time_rtc_min_delay=0, training_time_rtc_max_delay=10)
    server = LocalPolicyServer(policy=DummyFlowPolicy(), stats=make_stats(), config=config,
        ckpt_path='synthetic', host='127.0.0.1', port=0,
        device=torch.device('cpu'), use_fill_in=False)
    prefix = np.arange(14, dtype=np.float32).reshape(2, 7)
    request = {'observation/state': np.zeros(7, dtype=np.float32),
        'client/image_color_space': 'BGR', 'rtc/enabled': True,
        'rtc/inference_delay_steps': 2, 'rtc/prev_chunk_left_over': prefix,
        **{f'observation/images/{name}': np.zeros((2, 2, 3), dtype=np.uint8)
           for name in config['policy_config']['camera_names']}}

    def query(port):
        client = WebsocketOpenPIClient('127.0.0.1', port)
        try:
            validate_policy_metadata(client.server_metadata)
            response = client.infer(request)
            assert response['actions'].shape == (1, 50, 7)
            np.testing.assert_array_equal(response['actions'][0, :2], prefix)
            np.testing.assert_array_equal(response['actions'][0, 2:], 3.)
        finally:
            client._ws.close()

    async def run():
        async with websockets.asyncio.server.serve(server._handler, '127.0.0.1', 0,
                compression=None, max_size=None) as listener:
            await asyncio.wait_for(asyncio.to_thread(query, listener.sockets[0].getsockname()[1]), 10.)

    asyncio.run(run())
