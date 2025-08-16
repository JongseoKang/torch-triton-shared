#!/bin/bash

# Script to copy custom Python files to virtual environment's PyTorch Inductor and Triton
# Working directory: torch-triton-shared (with _inductor & triton subdirectories)
# Compatible with Ubuntu 24.04

set -e  # Exit on any error

# Configuration
VENV_PATH="$HOME/triton_shared/triton/.venv"
SOURCE_DIR="."  # torch-triton-shared directory
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
    echo ""
    echo "Expected directory structure:"
    echo "  torch-triton-shared/"
    echo "  ├── _inductor/          # Files to copy to torch/_inductor/"
    echo "  └── triton/             # Files to copy to triton/python/triton/"
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
        "__pycache__"
        "*.pyc"
        "*.pyo"
    )
    
    for pattern in "${ignore_patterns[@]}"; do
        if [[ "${basename,,}" == ${pattern,,} ]]; then
            return 0  # Should ignore
        fi
    done
    
    return 1  # Should not ignore
}

# Alternative function using rsync (more reliable)
copy_tree_rsync() {
    local source_dir=$1
    local target_dir=$2
    local description=$3
    
    if [ ! -d "$source_dir" ]; then
        print_status $YELLOW "⚠ Source directory not found: $source_dir"
        return
    fi
    
    print_status $GREEN "\nCopying $description..."
    print_status $YELLOW "  From: $source_dir"
    print_status $YELLOW "  To: $target_dir"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    # Use rsync to copy with exclusions
    if command -v rsync >/dev/null 2>&1; then
        rsync -av \
            --exclude='.gitignore' \
            --exclude='README*' \
            --exclude='*.md' \
            --exclude='*.txt' \
            --exclude='LICENSE*' \
            --exclude='CHANGELOG*' \
            --exclude='*.log' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='*.pyo' \
            "$source_dir/" "$target_dir/"
        print_status $GREEN "  ✓ Directory copied successfully"
    else
        # Fallback to the manual method if rsync not available
        copy_tree "$source_dir" "$target_dir" "$description"
    fi
}

# Function to copy directory tree with filtering
copy_tree() {
    local source_dir=$1
    local target_dir=$2
    local description=$3
    local copied_count=0
    
    if [ ! -d "$source_dir" ]; then
        print_status $YELLOW "⚠ Source directory not found: $source_dir"
        return
    fi
    
    print_status $GREEN "\nCopying $description..."
    print_status $YELLOW "  From: $source_dir"
    print_status $YELLOW "  To: $target_dir"
    
    # Create target directory structure
    mkdir -p "$target_dir"
    
    # Use process substitution instead of pipe to avoid subshell
    while IFS= read -r -d '' source_file; do
        # Skip ignored files
        if should_ignore_file "$source_file"; then
            log_verbose "Ignoring: $(basename "$source_file")"
            continue
        fi
        
        # Calculate relative path from source directory
        rel_path=${source_file#$source_dir/}
        target_file="$target_dir/$rel_path"
        
        # Create target subdirectory if needed
        target_subdir=$(dirname "$target_file")
        mkdir -p "$target_subdir"
        
        # Create backup if target exists
        if [ -f "$target_file" ]; then
            log_verbose "Creating backup: ${target_file}.backup"
            cp "$target_file" "${target_file}.backup"
        fi
        
        # Copy the file
        cp "$source_file" "$target_file"
        print_status $GREEN "  ✓ $(basename "$source_file") → $rel_path"
        log_verbose "    $source_file -> $target_file"
        ((copied_count++))
    done < <(find "$source_dir" -type f -print0)
    
    if [ $copied_count -eq 0 ]; then
        print_status $YELLOW "  No files copied from $source_dir"
    else
        print_status $BLUE "  Copied $copied_count files"
    fi
}

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    print_status $RED "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

print_status $GREEN "Starting installation process..."
print_status $YELLOW "Working directory: $SOURCE_DIR"
print_status $YELLOW "Virtual environment: $VENV_PATH"

# Verify source directories exist
if [ ! -d "$SOURCE_DIR/_inductor" ] && [ ! -d "$SOURCE_DIR/triton" ]; then
    print_status $RED "Error: Neither _inductor nor triton directories found in $SOURCE_DIR"
    print_status $YELLOW "Expected structure:"
    print_status $YELLOW "  $SOURCE_DIR/_inductor/"
    print_status $YELLOW "  $SOURCE_DIR/triton/"
    exit 1
fi

# Find PyTorch installation in the virtual environment
PYTORCH_PATH=$(find "$VENV_PATH" -name "torch" -type d -path "*/site-packages/torch" | head -1)

if [ -z "$PYTORCH_PATH" ]; then
    print_status $RED "Error: PyTorch installation not found in virtual environment"
    exit 1
fi

log_verbose "Found PyTorch at: $PYTORCH_PATH"

# Find Triton installation in the virtual environment
TRITON_PYTHON_PATH=$(find "$VENV_PATH" -name "triton" -type d -path "*/site-packages/triton" | head -1)

if [ -z "$TRITON_PYTHON_PATH" ]; then
    print_status $YELLOW "Warning: Triton installation not found in virtual environment"
    print_status $YELLOW "Will try alternative path: $VENV_PATH/../python/triton/"
    TRITON_PYTHON_PATH="$VENV_PATH/../python/triton"
    mkdir -p "$TRITON_PYTHON_PATH"
fi

log_verbose "Triton Python path: $TRITON_PYTHON_PATH"

# Define target directories
INDUCTOR_TARGET="$PYTORCH_PATH/_inductor"
TRITON_TARGET="$TRITON_PYTHON_PATH"

print_status $BLUE "\nTarget locations:"
print_status $YELLOW "  PyTorch Inductor: $INDUCTOR_TARGET"
print_status $YELLOW "  Triton Python: $TRITON_TARGET"

# Initialize counters
TOTAL_COPIED=0

# Part 1: Copy _inductor subdirectories to torch/_inductor
if [ -d "$SOURCE_DIR/_inductor" ]; then
    print_status $GREEN "\nProcessing PyTorch Inductor files..."
    
    # Copy codegen subdirectory
    if [ -d "$SOURCE_DIR/_inductor/codegen" ]; then
        copy_tree_rsync "$SOURCE_DIR/_inductor/codegen" "$INDUCTOR_TARGET/codegen" "Inductor codegen files"
    else
        print_status $YELLOW "⚠ _inductor/codegen not found"
    fi
    
    # Copy runtime subdirectory
    if [ -d "$SOURCE_DIR/_inductor/runtime" ]; then
        copy_tree_rsync "$SOURCE_DIR/_inductor/runtime" "$INDUCTOR_TARGET/runtime" "Inductor runtime files"
    else
        print_status $YELLOW "⚠ _inductor/runtime not found"
    fi
    
    # Copy any root-level files in _inductor (excluding subdirectories)
    if find "$SOURCE_DIR/_inductor" -maxdepth 1 -type f -name "*.py" | grep -q .; then
        print_status $GREEN "\nCopying root-level inductor files..."
        find "$SOURCE_DIR/_inductor" -maxdepth 1 -type f -name "*.py" | while read source_file; do
            if ! should_ignore_file "$source_file"; then
                filename=$(basename "$source_file")
                target_file="$INDUCTOR_TARGET/$filename"
                
                # Create backup if target exists
                if [ -f "$target_file" ]; then
                    log_verbose "Creating backup: ${target_file}.backup"
                    cp "$target_file" "${target_file}.backup"
                fi
                
                cp "$source_file" "$target_file"
                print_status $GREEN "  ✓ $filename → $filename"
                log_verbose "    $source_file -> $target_file"
            fi
        done
    fi
else
    print_status $YELLOW "⚠ _inductor directory not found, skipping PyTorch Inductor installation"
fi

# Part 2: Copy triton directory (sibling of _inductor) to triton/python/triton/
if [ -d "$SOURCE_DIR/triton" ]; then
    print_status $GREEN "\nProcessing Triton files (sibling directory)..."
    copy_tree_rsync "$SOURCE_DIR/triton" "$TRITON_TARGET" "Triton Python files with subdirectories"
else
    print_status $YELLOW "⚠ triton directory (sibling of _inductor) not found, skipping Triton installation"
fi

print_status $GREEN "\n✅ Installation completed!"
print_status $BLUE "\nInstallation summary:"
print_status $YELLOW "  Source: $SOURCE_DIR"
print_status $YELLOW "  Virtual env: $VENV_PATH"

if [ -d "$SOURCE_DIR/_inductor" ]; then
    print_status $GREEN "  ✓ PyTorch Inductor files installed"
fi

if [ -d "$SOURCE_DIR/triton" ]; then
    print_status $GREEN "  ✓ Triton Python files installed"
fi

print_status $YELLOW "\nNote: Backups of overwritten files were created with .backup extension"

print_status $BLUE "\nTo verify the installation:"
if [ -d "$INDUCTOR_TARGET" ]; then
    print_status $NC "  ls -la $INDUCTOR_TARGET"
fi
if [ -d "$TRITON_TARGET" ]; then
    print_status $NC "  ls -la $TRITON_TARGET"
fi

print_status $GREEN "\nInstallation process finished successfully!"