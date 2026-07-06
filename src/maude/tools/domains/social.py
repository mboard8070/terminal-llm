"""Domain-owned tool schemas."""

TOOL_NAMES = {"social_x_post_video", "social_post"}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "social_post",
            "description": "Post content to a social media platform using the running browser session. Requires "
            "browser_login('<platform>') first — keep the browser open after login. Instagram "
            "requires an image. TikTok requires a video file. X, LinkedIn, Facebook, Reddit, and "
            "Bluesky can attach images or videos when media_path/video_path/image_path is provided. "
            "For Reddit, first line of content is the title, rest is the body. IMPORTANT: When the "
            "user asks to post WITH an image or video, you MUST pass media_path, video_path, or "
            "image_path. Do not make a text-only post if media was requested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "enum": ["x", "linkedin", "facebook", "instagram", "reddit", "tiktok", "bluesky"],
                        "description": "Target platform to post to.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content of the post. For Reddit: first line is title, rest is body.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Path to image/video to attach. Use for any "
                        "platform when attaching media. Required "
                        "for Instagram images and TikTok videos; "
                        "also valid for X videos.",
                    },
                    "media_path": {
                        "type": "string",
                        "description": "Alias for image_path. Preferred for "
                        "X/LinkedIn/Facebook/Reddit/Bluesky media "
                        "attachments, including videos.",
                    },
                    "video_path": {
                        "type": "string",
                        "description": "Alias for image_path when the attachment "
                        "is a video, especially for X or TikTok.",
                    },
                    "expect_media": {
                        "type": "boolean",
                        "description": "Set true when the user explicitly "
                        "requested an image or video attachment; "
                        "the tool will fail instead of posting "
                        "text-only if no path is supplied.",
                    },
                    "subreddit": {
                        "type": "string",
                        "description": "Reddit only — subreddit to post to (e.g. 'python'). Omit for user profile.",
                    },
                },
                "required": ["platform", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "social_x_post_video",
            "description": "Post a video to X/Twitter using the running browser session. Use this instead of "
            "browser tools whenever the user asks to upload/post a video to X. If "
            "video_path/media_path/image_path is omitted, the latest shared MP4/MOV/WebM is attached "
            "automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The text content of the X post."},
                    "video_path": {
                        "type": "string",
                        "description": "Path to the video file. Optional; latest shared video is used if omitted.",
                    },
                    "media_path": {"type": "string", "description": "Alias for video_path."},
                    "image_path": {
                        "type": "string",
                        "description": "Alias for video_path, retained for compatibility.",
                    },
                },
                "required": ["content"],
            },
        },
    },
]


def schemas() -> list[dict]:
    return list(SCHEMAS)
