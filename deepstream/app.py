from pyservicemaker import Pipeline, Flow

pipeline = Pipeline("playback")
video_file = "/opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"
Flow(pipeline).capture([video_file]).render()
pipeline.start()
pipeline.wait()