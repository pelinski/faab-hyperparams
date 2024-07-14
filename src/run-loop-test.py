from pybela import Streamer
import asyncio


vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4', 'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']
streamer = Streamer()


def main():
    if streamer.connect():
        timestamps = {var: [] for var in vars}
        buffers = {var: [] for var in vars}
        async def callback(block):
            for idx, buffer in enumerate(block):
                var = buffer["name"]
                timestamps[var].append(buffer["buffer"]["ref_timestamp"])
                print(var, timestamps[var][-1])
                buffers[var].append(buffer["buffer"]["data"])

                if idx < 4:
                    length = len(buffer['buffer']['data'])
                    print("sent", var, "with",length, "bytes")
                    streamer.send_buffer(idx, 'f', length, buffer['buffer']['data'])
    
                    
        streamer.start_streaming(vars, on_block_callback=callback)
        asyncio.run(asyncio.sleep(60))
        streamer.stop_streaming()

main()