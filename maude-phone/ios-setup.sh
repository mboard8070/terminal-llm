#!/bin/bash
# MAUDE iOS Setup Script
# Run this on your Mac after cloning/copying the maude-phone project
#
# Prerequisites:
#   - Xcode installed (from App Store)
#   - Xcode Command Line Tools: xcode-select --install
#   - Node.js installed (brew install node)
#   - CocoaPods installed (sudo gem install cocoapods)

set -e
cd "$(dirname "$0")"

echo "=== MAUDE iOS Setup ==="
echo ""

# 1. Install iOS platform
echo "[1/6] Installing @capacitor/ios..."
npm install @capacitor/ios@^6.2.1

# 2. Add iOS platform
if [ -d "ios" ]; then
    echo "[2/6] iOS platform already exists, syncing..."
else
    echo "[2/6] Adding iOS platform..."
    npx cap add ios
fi

# 3. Copy gateway certificate
echo "[3/6] Copying gateway certificate..."
CERT_SRC=""
if [ -f "ios-assets/gateway_cert.pem" ]; then
    CERT_SRC="ios-assets/gateway_cert.pem"
elif [ -f "../certs/cert.pem" ]; then
    CERT_SRC="../certs/cert.pem"
fi

if [ -n "$CERT_SRC" ]; then
    cp "$CERT_SRC" ios/App/App/gateway_cert.pem
    echo "  Copied certificate from $CERT_SRC"
else
    echo "  WARNING: No gateway certificate found. Copy it manually to ios/App/App/gateway_cert.pem"
fi

# 4. Apply Info.plist patches
echo "[4/6] Patching Info.plist..."
PLIST="ios/App/App/Info.plist"
if [ -f "$PLIST" ]; then
    # Add permissions if not already present
    if ! grep -q "NSMicrophoneUsageDescription" "$PLIST"; then
        # Insert before closing </dict></plist>
        sed -i '' '/<\/dict>/i\
\t<key>NSMicrophoneUsageDescription</key>\
\t<string>MAUDE needs microphone access for voice chat</string>\
\t<key>NSCameraUsageDescription</key>\
\t<string>MAUDE needs camera access to capture images</string>\
\t<key>NSLocalNetworkUsageDescription</key>\
\t<string>MAUDE connects to your local Spark server</string>\
\t<key>NSAppTransportSecurity</key>\
\t<dict>\
\t\t<key>NSAllowsArbitraryLoads</key>\
\t\t<true/>\
\t\t<key>NSExceptionDomains</key>\
\t\t<dict>\
\t\t\t<key>server.tail00a82a.ts.net</key>\
\t\t\t<dict>\
\t\t\t\t<key>NSExceptionAllowsInsecureHTTPLoads</key>\
\t\t\t\t<true/>\
\t\t\t\t<key>NSTemporaryExceptionAllowsInsecureHTTPSLoads</key>\
\t\t\t\t<true/>\
\t\t\t\t<key>NSIncludesSubdomains</key>\
\t\t\t\t<true/>\
\t\t\t</dict>\
\t\t\t<key>100.66.49.48</key>\
\t\t\t<dict>\
\t\t\t\t<key>NSExceptionAllowsInsecureHTTPLoads</key>\
\t\t\t\t<true/>\
\t\t\t\t<key>NSTemporaryExceptionAllowsInsecureHTTPSLoads</key>\
\t\t\t\t<true/>\
\t\t\t\t<key>NSIncludesSubdomains</key>\
\t\t\t\t<true/>\
\t\t\t</dict>\
\t\t</dict>\
\t</dict>
' "$PLIST"
        echo "  Patched Info.plist with permissions and ATS exceptions"
    else
        echo "  Info.plist already patched"
    fi
else
    echo "  WARNING: Info.plist not found at $PLIST"
    echo "  Apply ios-patches/Info.plist.patch manually after 'npx cap add ios'"
fi

# 5. Copy app icons
echo "[5/6] Copying app icons..."
ICONSET="ios/App/App/Assets.xcassets/AppIcon.appiconset"
if [ -d "$ICONSET" ]; then
    # Copy icons to the asset catalog
    for size in 120 152 167 180 1024; do
        SRC="ios-assets/AppIcon-${size}.png"
        if [ -f "$SRC" ]; then
            cp "$SRC" "$ICONSET/"
            echo "  Copied AppIcon-${size}.png"
        fi
    done

    # Write the Contents.json for the icon set
    cat > "$ICONSET/Contents.json" << 'ICONJSON'
{
  "images": [
    {
      "filename": "AppIcon-120.png",
      "idiom": "iphone",
      "scale": "2x",
      "size": "60x60"
    },
    {
      "filename": "AppIcon-180.png",
      "idiom": "iphone",
      "scale": "3x",
      "size": "60x60"
    },
    {
      "filename": "AppIcon-152.png",
      "idiom": "ipad",
      "scale": "2x",
      "size": "76x76"
    },
    {
      "filename": "AppIcon-167.png",
      "idiom": "ipad",
      "scale": "2x",
      "size": "83.5x83.5"
    },
    {
      "filename": "AppIcon-1024.png",
      "idiom": "ios-marketing",
      "scale": "1x",
      "size": "1024x1024"
    }
  ],
  "info": {
    "author": "xcode",
    "version": 1
  }
}
ICONJSON
    echo "  Updated Contents.json"
else
    echo "  WARNING: Icon set directory not found. Run 'npx cap add ios' first."
fi

# 6. Sync and open
echo "[6/6] Syncing and opening Xcode..."
npx cap sync ios
npx cap open ios

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps in Xcode:"
echo "  1. Select your iPhone as the target device"
echo "  2. Go to Signing & Capabilities"
echo "  3. Select your personal Apple ID team"
echo "  4. Change the Bundle Identifier if needed (e.g., com.yourname.maude)"
echo "  5. Press Cmd+R to build and run"
echo ""
echo "NOTE: Free sideloaded apps expire after 7 days."
echo "      Use the PWA (Add to Home Screen in Safari) for a no-expiry alternative."
