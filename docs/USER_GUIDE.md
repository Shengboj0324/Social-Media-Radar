# User Guide - Social Media Radar

**Welcome to Social Media Radar!** 🎉

This guide will help you get started with the world's most powerful multi-channel intelligence aggregation system.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Connecting Platforms](#connecting-platforms)
3. [Downloading Media](#downloading-media)
4. [Customizing Output](#customizing-output)
5. [Security & Privacy](#security--privacy)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### What is Social Media Radar?

Social Media Radar is your personal intelligence desk that:
- ✅ Monitors 13+ platforms (Reddit, YouTube, TikTok, Facebook, Instagram, WeChat, NYTimes, etc.)
- ✅ Downloads videos and images automatically
- ✅ Summarizes content with AI
- ✅ Delivers daily briefings
- ✅ Keeps everything secure with military-grade encryption

### Quick Start (3 Steps)

1. **Create Account**
   ```bash
   POST /api/v1/auth/register
   {
     "email": "you@example.com",
     "password": "your_secure_password"
   }
   ```

2. **Connect Platforms** (Just click "Connect"!)
   - No API keys to copy
   - No configuration files
   - Just one click per platform

3. **Get Your Daily Brief**
   - Automatic daily digest
   - Customized to your interests
   - Available in 14 formats

---

## Connecting Platforms

### The Easy Way (Recommended) ✅

**No technical knowledge required!**

#### Step 1: View Available Platforms

```bash
GET /api/v1/platforms/
```

Response:
```json
[
  {
    "platform": "reddit",
    "name": "Reddit",
    "description": "Connect your Reddit account to monitor subreddits",
    "requires_oauth": true,
    "is_connected": false
  },
  {
    "platform": "youtube",
    "name": "YouTube",
    "description": "Connect YouTube to track channels and videos",
    "requires_oauth": true,
    "is_connected": false
  }
  // ... more platforms
]
```

#### Step 2: Click "Connect"

```bash
POST /api/v1/platforms/connect/reddit
```

Response:
```json
{
  "platform": "reddit",
  "authorization_url": "https://www.reddit.com/api/v1/authorize?...",
  "message": "Visit the URL to authorize Reddit"
}
```

#### Step 3: Visit the URL

- Click the authorization URL
- Log in to the platform
- Click "Allow" or "Authorize"
- **Done!** ✅

The system automatically:
- Receives the authorization
- Stores credentials securely (encrypted)
- Refreshes tokens automatically
- Notifies you of success

### Supported Platforms

| Platform | Authentication | Media Download | Status |
|----------|---------------|----------------|--------|
| **Reddit** | OAuth 2.0 | ✅ Images, Videos, GIFs | Ready |
| **YouTube** | OAuth 2.0 | ✅ Videos, Thumbnails | Ready |
| **TikTok** | OAuth 2.0 | ✅ Videos | Ready |
| **Facebook** | OAuth 2.0 | ✅ Images, Videos | Ready |
| **Instagram** | OAuth 2.0 | ✅ Images, Videos, Stories | Ready |
| **WeChat** | OAuth 2.0 | ✅ Articles | Ready |
| **NYTimes** | API Key | ✅ Images | Ready |
| **Google News** | None | ✅ Images | Ready |
| **WSJ** | None | ✅ Images | Ready |
| **ABC News** | None | ✅ Images | Ready |
| **Apple News** | None | ✅ Images | Ready |

---

## Downloading Media

### Automatic Media Download

**Videos and images are downloaded automatically!**

When you connect a platform, the system:
1. Monitors your feeds
2. Detects videos and images
3. Downloads them automatically
4. Stores them securely
5. Includes them in your digest

### Manual Media Download

#### Download a Video

```bash
POST /api/v1/media/download/video
{
  "url": "https://www.youtube.com/watch?v=...",
  "platform": "youtube",
  "quality": "1080p",
  "extract_audio": false
}
```

**Quality Options**:
- `2160p` - 4K Ultra HD
- `1080p` - Full HD (recommended)
- `720p` - HD
- `480p` - SD
- `360p` - Low
- `240p` - Mobile
- `auto` - Best available

Response:
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "platform": "youtube",
  "media_type": "video",
  "title": "Amazing Video",
  "duration": 300,
  "width": 1920,
  "height": 1080,
  "file_size": 52428800,
  "format": "mp4",
  "local_path": "/media/videos/123e4567.../video.mp4",
  "download_url": "https://cdn.example.com/..."
}
```

#### Download an Image

```bash
POST /api/v1/media/download/image
{
  "url": "https://example.com/image.jpg",
  "platform": "instagram",
  "convert_format": "jpeg",
  "max_width": 1920,
  "max_height": 1080
}
```

**Format Options**:
- `jpeg` - Best for photos
- `png` - Best for graphics
- `webp` - Best for web
- `gif` - Animated images
- `original` - Keep original format

#### Batch Download

```bash
POST /api/v1/media/download/batch
{
  "urls": [
    "https://youtube.com/watch?v=...",
    "https://youtube.com/watch?v=...",
    "https://youtube.com/watch?v=..."
  ],
  "platform": "youtube",
  "media_type": "video",
  "max_concurrent": 5
}
```

Downloads multiple files concurrently (default: 5 at a time).

---

## Customizing Output

### 14 Output Formats Available

1. **Markdown** - Clean, readable text
2. **HTML** - Rich web format
3. **JSON** - Structured data
4. **PDF** - Printable document
5. **Email** - Daily digest email
6. **Slack** - Team notifications
7. **Discord** - Community updates
8. **Twitter** - Social media posts
9. **Infographic** - Visual summary
10. **Video** - AI-generated video summary
11. **Audio/Podcast** - Listen on the go
12. **Interactive Dashboard** - Web interface
13. **Mobile Push** - Phone notifications
14. **SMS** - Text message summary

### Customize Your Digest

```bash
POST /api/v1/preferences
{
  "output_formats": ["email", "pdf", "video"],
  "delivery_time": "08:00",
  "timezone": "America/New_York",
  "topics": ["AI", "technology", "world news"],
  "platforms": ["reddit", "youtube", "nytimes"],
  "summary_style": "professional",  // or "casual", "expert", "eli5"
  "include_media": true,
  "max_items": 20
}
```

### AI Summary Styles

**Professional** (Default):
> "Today's top AI stories include OpenAI's latest model release, showing 40% improvement in reasoning tasks. Meanwhile, regulatory discussions continue in the EU..."

**Casual** (Friendly):
> "Hey! So OpenAI just dropped a new model and it's pretty awesome - 40% better at thinking through problems. Also, the EU is still figuring out AI rules..."

**Expert** (Technical):
> "OpenAI's GPT-5 demonstrates significant architectural improvements with 40% enhanced performance on MMLU benchmarks. EU AI Act implementation timeline remains uncertain..."

**ELI5** (Explain Like I'm 5):
> "Imagine if your smart robot friend got even smarter - that's what happened today! And some grown-ups are making rules about robots..."

---

## Security & Privacy

### Your Data is Safe

**Military-Grade Encryption**:
- AES-256-GCM for data at rest
- RSA-4096 for key encryption
- TLS 1.3 for data in transit
- Multi-layer credential protection

**Privacy Features**:
- Your credentials are encrypted
- Only you can access your data
- No data sharing with third parties
- GDPR compliant
- Right to deletion

### Security Best Practices

1. **Use a Strong Password**
   - Minimum 12 characters
   - Mix of letters, numbers, symbols
   - Unique to this service

2. **Enable MFA** (Multi-Factor Authentication)
   ```bash
   POST /api/v1/auth/mfa/enable
   ```

3. **Review Connected Platforms**
   ```bash
   GET /api/v1/platforms/status
   ```

4. **Disconnect Unused Platforms**
   ```bash
   DELETE /api/v1/platforms/disconnect/reddit
   ```

5. **Monitor Access Logs**
   ```bash
   GET /api/v1/audit/logs
   ```

---

## Troubleshooting

### Common Issues

**Q: Platform connection failed**
- A: Check that you authorized the app on the platform
- A: Verify your account has necessary permissions
- A: Try disconnecting and reconnecting

**Q: Media download failed**
- A: Check that the URL is accessible
- A: Verify the media is publicly available
- A: Check your storage quota

**Q: No digest received**
- A: Check your delivery preferences
- A: Verify email address is correct
- A: Check spam folder

**Q: Slow performance**
- A: Reduce number of monitored platforms
- A: Decrease max_items in preferences
- A: Disable media downloads temporarily

### Getting Help

- 📧 Email: support@socialmediaradar.com
- 💬 Discord: discord.gg/socialmediaradar
- 📖 Docs: docs.socialmediaradar.com
- 🐛 Issues: github.com/socialmediaradar/issues

---

## Next Steps

1. ✅ Connect your first platform
2. ✅ Customize your preferences
3. ✅ Receive your first digest
4. ✅ Download some media
5. ✅ Explore different output formats

**Welcome to the future of information aggregation!** 🚀

