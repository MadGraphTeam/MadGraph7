if [ ! -d "src" ] && [ ! -d "SubProcesses" ]; then
  echo "Error: Neither 'src' nor 'SubProcesses' directory exists, script needs to be run from the root directory of a generated process" >&2
  exit 255
fi

count=0
total=0
files_needing_format=()
patch_file="format-changes.patch"
num_file="files2format"

# clean patch file if it exists
> "$patch_file"
> "$num_file"

echo ""
echo "Checking formatting with clang-format..."
echo ""

while IFS= read -r file; do
  ((total++))
  if ! clang-format --dry-run --Werror "$file" 2>/dev/null; then
    echo "=== $file needs formatting ==="
    clang-format "$file" | diff -u "$file" - >> "$patch_file"
    echo "" >> "$patch_file"
    ((count++))
    files_needing_format+=("$file")
  fi
done < <(find src SubProcesses -type f \( -name "*.cc" -o -name "*.h" \) 2>/dev/null)


echo ""
echo "Files needing formatting: $count"
echo "Total files checked: $total"

if [ $count -gt 0 ]; then
  echo "Detailed patch saved to '$patch_file'"
  echo "If this scrpt is run in the CI workflow, detailed patches are provided as comment on the PR." 
else
  echo "All files are properly formatted."
  rm "$patch_file"  
fi
echo ""

echo $count | tr -d \\n >> $num_file
