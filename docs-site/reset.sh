#!/bin/bash
print_status() {
    echo -e "\033[1;34m>>> $1\033[0m"
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR" || exit 1

print_status "Current directory: $(pwd)"

if [ "$1" = "--full" ]; then
    print_status "Running full reset (reinstalling node_modules)..."
    
    print_status "Stopping any running Next.js processes..."
    pkill -f "next"
    
    print_status "Clearing Next.js cache..."
    rm -rf .next
    
    print_status "Removing node_modules and package-lock.json..."
    rm -rf node_modules package-lock.json
    
    print_status "Clearing npm cache..."
    npm cache clean --force
    
    print_status "Reinstalling dependencies..."
    npm install
    
    print_status "Building project..."
    npm run build
    
    print_status "Starting development server..."
    npm run dev
else
    print_status "Running standard reset (keeping node_modules)..."
    
    print_status "Stopping any running Next.js processes..."
    pkill -f "next"
    
    print_status "Clearing Next.js cache..."
    rm -rf .next
    
    print_status "Building project..."
    npm run build
    
    print_status "Starting development server..."
    npm run dev
fi
