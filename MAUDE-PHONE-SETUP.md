# MAUDE Phone — Samsung Android Setup

## Phase 1: PWA Installation (No Root Required)

1. Connect phone to Tailscale VPN
2. Open Samsung Internet browser
3. Navigate to `http://spark-e26c:30000/`
4. Tap browser menu (three dots) → "Add page to" → "Home screen"
5. MAUDE launches in standalone mode (no browser chrome)
6. In MAUDE Settings, select "80s Green CRT" or "80s Amber CRT" theme

## Phase 2: Root & Debloat Samsung

### Prerequisites
- Samsung Galaxy phone (USB debugging enabled)
- PC with ADB/Fastboot installed
- USB cable

### Step 1: Unlock Bootloader
```bash
# On the phone:
# Settings → About Phone → Software Information → tap "Build Number" 7 times
# Settings → Developer Options → Enable "OEM Unlocking"
# Settings → Developer Options → Enable "USB Debugging"

# Reboot to download mode:
# Power off, then hold: Volume Down + Power (or Volume Down + Bixby + Power)
# Long press Volume Up to enter Download Mode
# WARNING: This will factory reset the device
```

### Step 2: Flash TWRP / Magisk
```bash
# Download latest Magisk APK from https://github.com/topjohnwu/Magisk/releases
# Download your device's stock firmware (use Frija or SamFirm)
# Extract AP tar, patch boot.img with Magisk app on another device

# Flash patched boot via Odin (Windows) or Heimdall (Linux):
heimdall flash --BOOT magisk_patched_boot.img

# After reboot, install Magisk APK
adb install Magisk-v28.0.apk
```

### Step 3: Debloat via ADB (No Root Needed for This Step)
```bash
# Connect via USB, verify:
adb devices

# Remove Samsung bloatware (disables, doesn't delete — reversible):
adb shell pm uninstall -k --user 0 com.samsung.android.app.spage        # Samsung Free
adb shell pm uninstall -k --user 0 com.samsung.android.bixby.agent      # Bixby Voice
adb shell pm uninstall -k --user 0 com.samsung.android.bixby.service    # Bixby Service
adb shell pm uninstall -k --user 0 com.samsung.android.visionintelligence  # Bixby Vision
adb shell pm uninstall -k --user 0 com.samsung.android.game.gamehome    # Game Launcher
adb shell pm uninstall -k --user 0 com.samsung.android.game.gametools   # Game Tools
adb shell pm uninstall -k --user 0 com.samsung.android.app.tips         # Tips
adb shell pm uninstall -k --user 0 com.samsung.android.mobileservice    # Samsung Experience
adb shell pm uninstall -k --user 0 com.samsung.android.app.routines     # Routines
adb shell pm uninstall -k --user 0 com.samsung.android.ardrawing        # AR Doodle
adb shell pm uninstall -k --user 0 com.samsung.android.aremoji          # AR Emoji
adb shell pm uninstall -k --user 0 com.samsung.android.arzone           # AR Zone
adb shell pm uninstall -k --user 0 com.samsung.android.app.dressroom    # Samsung Shop
adb shell pm uninstall -k --user 0 com.samsung.android.app.watchmanagerstub  # Galaxy Wearable
adb shell pm uninstall -k --user 0 com.samsung.android.livestickers     # Live Stickers
adb shell pm uninstall -k --user 0 com.samsung.android.app.social       # What's New
adb shell pm uninstall -k --user 0 com.sec.android.app.sbrowser         # Samsung Internet (use MAUDE)
adb shell pm uninstall -k --user 0 com.samsung.android.email.provider   # Samsung Email
adb shell pm uninstall -k --user 0 com.samsung.android.calendar         # Samsung Calendar
adb shell pm uninstall -k --user 0 com.samsung.android.app.reminder     # Reminder
adb shell pm uninstall -k --user 0 com.microsoft.skydrive               # OneDrive
adb shell pm uninstall -k --user 0 com.facebook.katana                  # Facebook
adb shell pm uninstall -k --user 0 com.facebook.orca                    # Messenger
adb shell pm uninstall -k --user 0 com.facebook.services                # Facebook Services
adb shell pm uninstall -k --user 0 com.linkedin.android                 # LinkedIn
adb shell pm uninstall -k --user 0 com.spotify.music                    # Spotify
adb shell pm uninstall -k --user 0 com.netflix.mediaclient              # Netflix
adb shell pm uninstall -k --user 0 com.samsung.android.themestore       # Galaxy Themes

# Keep these:
# com.android.chrome (fallback browser)
# com.tailscale.ipn (Tailscale)
# com.android.dialer (Phone)
# com.android.settings (System Settings)
```

### Step 4: Set MAUDE as Default Launcher
```bash
# Option A: Install a minimal launcher (e.g., KISS Launcher or Olauncher)
# and set it as default, with only MAUDE PWA shortcut visible

# Option B: With root + ADB, force MAUDE PWA as the home app:
# Install Hermit (lite apps) or WebView wrapper to create a launchable app from the PWA
# Then set as default launcher:
adb shell cmd package set-home-activity com.example.maude/.MainActivity

# Option C: Install a custom minimal launcher with only these apps:
# - MAUDE (PWA shortcut)
# - Tailscale
# - Phone
# - Settings
```

### Step 5: Retro Boot Animation (Requires Root)
```bash
# Create a retro boot animation (green/amber text on black):
# Format: bootanimation.zip containing desc.txt + part0/ + part1/

# desc.txt:
# 1080 2400 30
# p 1 0 part0
# p 0 0 part1

# part0/: MAUDE splash frames (green text "MAUDE SYSTEM LOADING" on black)
# part1/: Matrix-style cascade or blinking cursor

# Push to device:
adb root
adb remount
adb push bootanimation.zip /system/media/bootanimation.zip
adb shell chmod 644 /system/media/bootanimation.zip
```

### Step 6: Custom Wallpaper
```bash
# Set a solid black wallpaper or retro green grid:
adb shell am start -a android.intent.action.ATTACH_DATA \
  -t image/png \
  -d file:///sdcard/maude_wallpaper.png \
  com.android.wallpaperpicker
```

## Essential Apps to Keep

| App | Package | Why |
|-----|---------|-----|
| Tailscale | `com.tailscale.ipn` | VPN to Spark |
| Phone | `com.android.dialer` | Calls |
| Settings | `com.android.settings` | WiFi, etc |
| Chrome | `com.android.chrome` | Fallback browser |
| MAUDE PWA | (home screen shortcut) | The main event |

Everything else goes.

## Theme: 80s Terminal Look

The PWA has built-in retro themes:
- **80s Green CRT**: Phosphor green on black, scanlines, CRT flicker
- **80s Amber CRT**: Amber phosphor on black, scanlines, CRT flicker

Select in Settings → Theme. The whole app transforms — chat bubbles, home screen, terminal, everything.

For maximum 80s vibes:
1. Set wallpaper to solid black
2. Enable "80s Green CRT" in MAUDE Settings
3. Use the Terminal app with its matching green/amber terminal theme
4. Disable Samsung's navigation gesture hints
   ```
   adb shell settings put global navigation_bar_gesture_hint 0
   ```
5. Force dark mode system-wide
   ```
   adb shell settings put secure ui_night_mode 2
   ```
