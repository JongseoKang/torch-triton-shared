#!/bin/bash

# Script to copy custom Python files to virtual environment's PyTorch Inductor, Triton, and TritonShared
# Working directory: torch-triton-shared (with _inductor, triton & triton_shared subdirectories)
# Compatible with Ubuntu 24.04

set -e  # Exit on any error

# Configuration
VENV_PATH="$HOME/triton_shared/triton/.venv"
TRITON_SHARED_PATH="$HOME/triton_shared"
TRITON_PYTHON_PATH="$HOME/triton_shared/triton/python/triton/"
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
    echo "  -v, --verbose         Enable verbose output"
    echo "  -s, --source          Source directory (default: current directory)"
    echo "  -e, --venv            Virtual environment path (default: \$HOME/triton_shared/triton/.venv)"
    echo "  -t, --triton-shared   TritonShared target path (default: \$HOME/triton_shared)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Expected directory structure:"
    echo "  torch-triton-shared/"
    echo "  ├── _inductor/          # Files to copy to torch/_inductor/"
    echo "  ├── triton/             # Files to copy to triton/python/triton/"
    echo "  └── triton_shared/      # Files to copy to \$HOME/triton_shared/"
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
        -t|--triton-shared)
            TRITON_SHARED_PATH="$2"
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
        ".git"
        ".gitmodules"
        "*.backup"
        # NOTE: CMakeLists.txt is NOT ignored - we want to copy it
    )
    
    for pattern in "${ignore_patterns[@]}"; do
        if [[ "${basename,,}" == ${pattern,,} ]]; then
            return 0  # Should ignore
        fi
    done
    
    return 1  # Should not ignore
}

# Alternative function using rsync (more reliable) - ADDITIVE COPY ONLY
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
    
    # Create target directory (DO NOT REMOVE EXISTING)
    mkdir -p "$target_dir"
    
    # Use rsync to copy with exclusions - ADDITIVE COPY ONLY (NO --delete)
    if command -v rsync >/dev/null 2>&1; then
        rsync -av \
            --exclude='.gitignore' \
            --exclude='README*' \
            --exclude='*.md' \
            --exclude='license*' \
            --exclude='LICENSE*' \
            --exclude='CHANGELOG*' \
            --exclude='*.log' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='*.pyo' \
            --exclude='.git*' \
            --exclude='*.backup' \
            --include='CMakeLists.txt' \
            --include='*.cmake' \
            --include='*.cpp' \
            --include='*.h' \
            --include='*.hpp' \
            --include='*.py' \
            --include='*.td' \
            --include='*.inc' \
            "$source_dir/" "$target_dir/"
        print_status $GREEN "  ✓ Directory copied additively (preserving existing files)"
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
        # Skip ignored files (but allow CMakeLists.txt and build files)
        if should_ignore_file "$source_file"; then
            # Special exception for important build files
            local basename=$(basename "$source_file")
            if [[ "${basename,,}" != "cmakelists.txt" ]] && [[ "${basename}" != *.cmake ]] && [[ "${basename}" != *.cpp ]] && [[ "${basename}" != *.h ]] && [[ "${basename}" != *.hpp ]] && [[ "${basename}" != *.py ]] && [[ "${basename}" != *.td ]] && [[ "${basename}" != *.inc ]]; then
                log_verbose "Ignoring: $(basename "$source_file")"
                continue
            fi
        fi
        
        # Calculate relative path from source directory
        rel_path=${source_file#$source_dir/}
        target_file="$target_dir/$rel_path"
        
        # Create target subdirectory if needed
        target_subdir=$(dirname "$target_file")
        mkdir -p "$target_subdir"
        
        # Copy the file (no backup)
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

# Validation function
validate_paths() {
    print_status $BLUE "Validating paths..."
    
    # Check if source directory exists
    if [ ! -d "$SOURCE_DIR" ]; then
        print_status $RED "Error: Source directory not found: $SOURCE_DIR"
        exit 1
    fi
    
    # Check if virtual environment exists (only if we're copying to it)
    if [ -d "$SOURCE_DIR/_inductor" ] || [ -d "$SOURCE_DIR/triton" ]; then
        if [ ! -d "$VENV_PATH" ]; then
            print_status $RED "Error: Virtual environment not found at $VENV_PATH"
            print_status $YELLOW "Hint: Make sure your virtual environment is set up correctly"
            exit 1
        fi
    fi
    
    # Create triton_shared target directory if it doesn't exist
    if [ -d "$SOURCE_DIR/triton_shared" ]; then
        mkdir -p "$TRITON_SHARED_PATH"
        log_verbose "TritonShared target directory: $TRITON_SHARED_PATH"
    fi
}

print_status $GREEN "Starting installation process..."
print_status $YELLOW "Working directory: $SOURCE_DIR"
print_status $YELLOW "Virtual environment: $VENV_PATH"
print_status $YELLOW "TritonShared target: $TRITON_SHARED_PATH"

# Validate paths
validate_paths

# Verify source directories exist
has_inductor=$([ -d "$SOURCE_DIR/_inductor" ] && echo "true" || echo "false")
has_triton=$([ -d "$SOURCE_DIR/triton" ] && echo "true" || echo "false")
has_triton_shared=$([ -d "$SOURCE_DIR/triton_shared" ] && echo "true" || echo "false")

if [ "$has_inductor" = "false" ] && [ "$has_triton" = "false" ] && [ "$has_triton_shared" = "false" ]; then
    print_status $RED "Error: No valid source directories found in $SOURCE_DIR"
    print_status $YELLOW "Expected structure:"
    print_status $YELLOW "  $SOURCE_DIR/_inductor/"
    print_status $YELLOW "  $SOURCE_DIR/triton/"
    print_status $YELLOW "  $SOURCE_DIR/triton_shared/"
    exit 1
fi

print_status $BLUE "\nFound source directories:"
[ "$has_inductor" = "true" ] && print_status $GREEN "  ✓ _inductor/"
[ "$has_triton" = "true" ] && print_status $GREEN "  ✓ triton/"
[ "$has_triton_shared" = "true" ] && print_status $GREEN "  ✓ triton_shared/"

# Part 1: Copy _inductor subdirectories to torch/_inductor
if [ "$has_inductor" = "true" ]; then
    # Use fixed PyTorch path in virtual environment
    PYTORCH_PATH=$(find "$VENV_PATH" -name "torch" -type d -path "*/site-packages/torch" | head -1)

    if [ -z "$PYTORCH_PATH" ]; then
        print_status $RED "Error: PyTorch installation not found in virtual environment at $VENV_PATH"
        print_status $YELLOW "Skipping PyTorch Inductor installation"
    else
        log_verbose "Found PyTorch at: $PYTORCH_PATH"
        INDUCTOR_TARGET="$PYTORCH_PATH/_inductor"
        
        print_status $GREEN "\nProcessing PyTorch Inductor files..."
        print_status $YELLOW "  Target: $INDUCTOR_TARGET"
        
        # Copy the entire _inductor directory completely
        copy_tree_rsync "$SOURCE_DIR/_inductor" "$INDUCTOR_TARGET" "All Inductor files and subdirectories"
    fi
fi

# Part 2: Copy triton directory to fixed triton path
if [ "$has_triton" = "true" ]; then
    print_status $GREEN "\nProcessing Triton files..."
    print_status $YELLOW "  Target: $TRITON_PYTHON_PATH"
    
    # Copy the entire triton directory completely
    copy_tree_rsync "$SOURCE_DIR/triton" "$TRITON_PYTHON_PATH" "All Triton Python files with subdirectories"
fi

# Part 3: Copy triton_shared directory to fixed triton_shared path
if [ "$has_triton_shared" = "true" ]; then
    print_status $GREEN "\nProcessing TritonShared files..."
    print_status $YELLOW "  Target: $TRITON_SHARED_PATH"
    
    # Copy the entire triton_shared directory completely
    copy_tree_rsync "$SOURCE_DIR/triton_shared" "$TRITON_SHARED_PATH" "All TritonShared files with subdirectories"
    
    # Verify TritonShared installation
    if [ -d "$TRITON_SHARED_PATH" ]; then
        print_status $GREEN "  ✓ TritonShared files installed successfully"
        log_verbose "TritonShared contents:"
        if [ "$VERBOSE" = "true" ]; then
            find "$TRITON_SHARED_PATH" -type f -name "*.py" | head -10 | while read file; do
                rel_path=${file#$TRITON_SHARED_PATH/}
                log_verbose "    $rel_path"
            done
        fi
    fi
fi

print_status $GREEN "\n✅ Installation completed!"
print_status $BLUE "\nInstallation summary:"
print_status $YELLOW "  Source: $SOURCE_DIR"

if [ "$has_inductor" = "true" ] && [ -n "$PYTORCH_PATH" ]; then
    print_status $GREEN "  ✓ PyTorch Inductor files completely copied to: $PYTORCH_PATH/_inductor"
fi

if [ "$has_triton" = "true" ]; then
    print_status $GREEN "  ✓ Triton Python files completely copied to: $TRITON_PYTHON_PATH"
fi

if [ "$has_triton_shared" = "true" ]; then
    print_status $GREEN "  ✓ TritonShared files completely copied to: $TRITON_SHARED_PATH"
fi

print_status $YELLOW "\nNote: Additive copy performed - existing files overwritten without backup"

print_status $BLUE "\nTo verify the installation:"
if [ "$has_inductor" = "true" ] && [ -n "$PYTORCH_PATH" ]; then
    print_status $NC "  ls -la $PYTORCH_PATH/_inductor"
fi
if [ "$has_triton" = "true" ]; then
    print_status $NC "  ls -la $TRITON_PYTHON_PATH"
fi
if [ "$has_triton_shared" = "true" ]; then
    print_status $NC "  ls -la $TRITON_SHARED_PATH"
fi

print_status $GREEN "\nInstallation process finished successfully!"