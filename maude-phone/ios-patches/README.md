# MAUDE iOS Setup Guide

## Option A: PWA (Recommended for daily use — no expiry)

The easiest way to run MAUDE on your iPhone. No Xcode, no signing, no 7-day expiry.

1. Open **Safari** on your iPhone
2. Navigate to `https://100.107.132.16:30000`
3. Accept the self-signed certificate warning
4. Tap the **Share** button (square with arrow)
5. Tap **"Add to Home Screen"**
6. Tap **Add**

MAUDE will appear as a full-screen app on your home screen. It works as long as the Spark server is reachable on your network (Tailscale).

### Limitations
- No native mic/camera access beyond what Safari allows
- Must be on the same Tailscale network as Spark
- First load requires network — subsequent loads use cached app shell

---

## Option B: Capacitor Native App (sideload via Xcode)

Full native capabilities (mic, camera, WebSocket). Requires a Mac with Xcode.

### Prerequisites

- **Mac** with macOS 13+
- **Xcode 15+** (free from App Store)
- **Xcode Command Line Tools**: `xcode-select --install`
- **Node.js 18+**: `brew install node`
- **CocoaPods**: `sudo gem install cocoapods`
- **Apple ID** (free, no $99 developer account needed)

### Quick Setup (Automated)

```bash
# 1. Transfer the maude-phone project to your Mac
# 2. Install dependencies
cd maude-phone
npm install

# 3. Run the setup script
chmod +x ios-setup.sh
./ios-setup.sh
```

The script will:
- Install `@capacitor/ios`
- Add the iOS platform
- Copy the gateway certificate
- Patch Info.plist with permissions and ATS exceptions
- Copy app icons
- Sync and open Xcode

### Manual Setup

If the automated script doesn't work:

```bash
cd maude-phone
npm install
npm install @capacitor/ios@^6.2.1
npx cap add ios
npx cap sync ios
```

Then manually:

1. Copy `ios-patches/Info.plist.patch` entries into `ios/App/App/Info.plist`
2. Copy icon PNGs from `ios-assets/` into `ios/App/App/Assets.xcassets/AppIcon.appiconset/`
3. Open in Xcode: `npx cap open ios`

### Building in Xcode

1. Select your **iPhone** as the target device (connect via USB or Wi-Fi)
2. Go to **Signing & Capabilities** tab
3. Set **Team** to your personal Apple ID
4. Change **Bundle Identifier** to something unique (e.g., `com.yourname.maude`)
5. Press **Cmd+R** to build and run

### Important Notes

- **7-day expiry**: Free sideloaded apps must be re-deployed every 7 days
  - Just press Cmd+R in Xcode again to refresh
  - Your data (conversations, settings) is preserved between refreshes
- **3 app limit**: Free Apple IDs can only have 3 sideloaded apps at a time
- **Same network**: Your iPhone must be on the same Tailscale network as Spark
- **Self-signed cert**: The ATS exception in Info.plist allows the self-signed certificate

### Troubleshooting

| Issue | Fix |
|-------|-----|
| "Untrusted Developer" on iPhone | Settings → General → VPN & Device Management → Trust your profile |
| Build fails with signing error | Change Bundle Identifier to a unique value |
| Can't connect to Spark | Check Tailscale is connected on both devices |
| White screen after launch | Check server.url in capacitor.config.ts matches your Spark IP |
| "App is no longer available" | Re-deploy from Xcode (7-day expiry) |
