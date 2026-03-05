#!/bin/bash
# Generate all required PNG icon sizes from the SVG source
# Requires: cairosvg (pip install cairosvg) or rsvg-convert (librsvg2-bin)

set -e
cd "$(dirname "$0")"

SVG="public/assets/maude-icon.svg"
SIZES=(120 152 167 180 512 1024)

if command -v cairosvg &>/dev/null; then
    for size in "${SIZES[@]}"; do
        cairosvg "$SVG" -o "public/assets/maude-icon-${size}.png" -W "$size" -H "$size"
        echo "Generated maude-icon-${size}.png"
    done
elif command -v rsvg-convert &>/dev/null; then
    for size in "${SIZES[@]}"; do
        rsvg-convert -w "$size" -h "$size" "$SVG" > "public/assets/maude-icon-${size}.png"
        echo "Generated maude-icon-${size}.png"
    done
else
    echo "Error: Need cairosvg or rsvg-convert to generate icons"
    echo "  pip install cairosvg"
    echo "  brew install librsvg  (macOS)"
    exit 1
fi

# Copy icons to ios-assets/ for the bundle
mkdir -p ios-assets
for size in "${SIZES[@]}"; do
    cp "public/assets/maude-icon-${size}.png" "ios-assets/AppIcon-${size}.png"
done
echo "Copied to ios-assets/"

echo "Done! All icons generated."
