#!/bin/bash

# Script to copy custom Python files to virtual environment's PyTorch Inductor
# Compatible with Ubuntu 24.04

set -e  # Exit on any error

# Configuration
VENV_PATH="triton_shared/triton/.venv"
SOURCE_DIR="."  # Current directory, change if needed
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -v, --verbose    Enable verbose output"
    echo "  -s, --source     Source directory (default: current directory)"
    echo "  -e, --venv       Virtual environment path (default: triton_shared/triton/.venv)"
    echo "  -h, --help       Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -s|--source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -e|--venv)
            VENV_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_status $RED "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function for verbose logging
log_verbose() {
    if [ "$VERBOSE" = true ]; then
        print_status $BLUE "  $1"
    fi
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_status $RED "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

print_status $GREEN "Starting file copy process..."
print_status $YELLOW "Virtual environment: $VENV_PATH"
print_status $YELLOW "Source directory: $SOURCE_DIR"

# Find PyTorch installation in the virtual environment
PYTORCH_PATH=$(find "$VENV_PATH" -name "torch" -type d -path "*/site-packages/torch" | head -1)

if [ -z "$PYTORCH_PATH" ]; then
    print_status $RED "Error: PyTorch installation not found in virtual environment"
    exit 1
fi

log_verbose "Found PyTorch at: $PYTORCH_PATH"

# Define target directories
INDUCTOR_TARGET="$PYTORCH_PATH/_inductor"
TRITON_CPU_TARGET="$VENV_PATH/lib/python*/site-packages/triton/language/extra/cpu"

# Expand the triton path wildcard
TRITON_CPU_TARGET=$(echo $TRITON_CPU_TARGET)

print_status $GREEN "Target directories:"
print_status $YELLOW "  Inductor: $INDUCTOR_TARGET"
print_status $YELLOW "  Triton CPU: $TRITON_CPU_TARGET"

# Create target directories if they don't exist
mkdir -p "$INDUCTOR_TARGET"
if [ -n "$TRITON_CPU_TARGET" ] && [ -d "$(dirname "$TRITON_CPU_TARGET")" ]; then
    mkdir -p "$TRITON_CPU_TARGET"
fi

# Counter for copied files
COPIED_COUNT=0

# Function to check if file should be ignored
should_ignore_file() {
    local file=$1
    local basename=$(basename "$file")
    
    # List of patterns to ignore (case-insensitive)
    local ignore_patterns=(
        ".gitignore"
        "readme*"
        "*.md"
        "*.txt"
        "license*"
        "changelog*"
        "*.log"
    )
    
    for pattern in "${ignore_patterns[@]}"; do
        if [[ "${basename,,}" == ${pattern,,} ]]; then
            return 0  # Should ignore
        fi
    done
    
    return 1  # Should not ignore
}

# Function to copy file with backup
copy_file() {
    local source_file=$1
    local target_file=$2
    local description=$3
    
    # Check if file should be ignored
    if should_ignore_file "$source_file"; then
        log_verbose "Ignoring file: $(basename "$source_file")"
        return
    fi
    
    if [ -f "$source_file" ]; then
        # Create backup if target exists
        if [ -f "$target_file" ]; then
            log_verbose "Creating backup: ${target_file}.backup"
            cp "$target_file" "${target_file}.backup"
        fi
        
        # Copy the file
        cp "$source_file" "$target_file"
        print_status $GREEN "✓ Copied $description"
        log_verbose "  $source_file -> $target_file"
        ((COPIED_COUNT++))
    else
        print_status $YELLOW "⚠ Source file not found: $source_file"
    fi
}

print_status $BLUE "\nCopying files..."

# 1. Copy _*inductor files (from runtime, codegen, or root)
print_status $YELLOW "\n1. Processing _*inductor files..."

# Find all directories matching _*inductor pattern
find "$SOURCE_DIR" -type d -name "_*inductor" 2>/dev/null | while read inductor_dir; do
    log_verbose "Processing directory: $inductor_dir"
    
    # Copy files from runtime subdirectory
    if [ -d "$inductor_dir/runtime" ]; then
        find "$inductor_dir/runtime" -name "*.py" -type f | while read file; do
            filename=$(basename "$file")
            copy_file "$file" "$INDUCTOR_TARGET/$filename" "inductor runtime file: $filename"
        done
    fi
    
    # Copy files from codegen subdirectory
    if [ -d "$inductor_dir/codegen" ]; then
        find "$inductor_dir/codegen" -name "*.py" -type f | while read file; do
            filename=$(basename "$file")
            copy_file "$file" "$INDUCTOR_TARGET/$filename" "inductor codegen file: $filename"
        done
    fi
    
    # Copy files directly from inductor directory (excluding subdirectories)
    find "$inductor_dir" -maxdepth 1 -name "*.py" -type f | while read file; do
        filename=$(basename "$file")
        copy_file "$file" "$INDUCTOR_TARGET/$filename" "inductor file: $filename"
    done
done

# 2. Copy libdevice.py for triton CPU
print_status $YELLOW "\n2. Processing Triton CPU files..."

# Find libdevice.py files
find "$SOURCE_DIR" -name "libdevice.py" -type f 2>/dev/null | while read file; do
    if [ -d "$TRITON_CPU_TARGET" ]; then
        copy_file "$file" "$TRITON_CPU_TARGET/libdevice.py" "Triton CPU libdevice.py"
    else
        print_status $YELLOW "⚠ Triton CPU target directory not found, skipping libdevice.py"
    fi
done

# 3. Copy __init__.py files for triton CPU
find "$SOURCE_DIR" -name "__init__.py" -path "*/cpu/*" -type f 2>/dev/null | while read file; do
    if [ -d "$TRITON_CPU_TARGET" ]; then
        copy_file "$file" "$TRITON_CPU_TARGET/__init__.py" "Triton CPU __init__.py"
    else
        print_status $YELLOW "⚠ Triton CPU target directory not found, skipping __init__.py"
    fi
done

# Also look for general __init__.py files if the specific path doesn't exist
if [ ! -f "$TRITON_CPU_TARGET/__init__.py" ]; then
    find "$SOURCE_DIR" -name "__init__.py" -type f 2>/dev/null | head -1 | while read file; do
        if [ -d "$TRITON_CPU_TARGET" ]; then
            copy_file "$file" "$TRITON_CPU_TARGET/__init__.py" "Triton CPU __init__.py (general)"
        fi
    done
fi

print_status $GREEN "\n✅ Copy process completed!"
print_status $BLUE "Files processed. Check the output above for details."

# Note: The COPIED_COUNT variable won't be accurate due to subshells in while loops
# This is a limitation of bash when using pipes with while loops

print_status $YELLOW "\nNote: If you need to restore original files, backups were created with .backup extension"
print_status $BLUE "\nTo verify the installation, you can check the target directories:"
print_status $NC "  ls -la $INDUCTOR_TARGET"
if [ -d "$TRITON_CPU_TARGET" ]; then
    print_status $NC "  ls -la $TRITON_CPU_TARGET"
fi