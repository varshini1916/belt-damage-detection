from video_inspection import process_video

input_video = "test_videos/conveyor_test.mp4"
output_video = "test_videos/conveyor_result.mp4"

result = process_video(
    input_video=input_video,
    output_video=output_video,
    confidence=0.24,
    imgsz=640
)

print("\n==============================")
print("BELTGUARD VIDEO INSPECTION")
print("==============================")

for key, value in result.items():
    print(f"{key}: {value}")