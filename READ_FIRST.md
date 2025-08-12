#  READ FIRST - Daily Development Reference

Welcome to the Lumina Memory System! This guide covers your daily development workflow.

## 🚀 **Daily Development Commands**

### **Starting a New Feature**
`ash
# 1. Always start from a clean main branch
git checkout main
git pull origin main

# 2. Create a feature branch for each change
git checkout -b feature/your-feature-name
`

### **Working on Your Feature**
`ash
# 3. Make changes, test locally
# ... edit files ...

# 4. Check what changed
git status
git diff

# 5. Stage and commit incremental changes
git add specific_files.py
git commit -m "feat: Add specific improvement"

# 6. Push feature branch to GitHub
git push -u origin feature/your-feature-name
`

### **After Feature is Complete**
`ash
# 7. Create Pull Request on GitHub web interface
# 8. CI runs automatically on PR
# 9. Review and merge when ready
# 10. Cleanup after merge
git checkout main
git pull origin main
git branch -d feature/your-feature-name  # Delete local branch
`

##  **Development vs Production**

### **Development Environment (Feature Branches):**
-  Your playground - can experiment freely
-  Break things, try new ideas, make mistakes
-  Fast iteration cycle
-  Only affects you

### **Production Environment (Main Branch):**
-  Stable, tested, reliable
-  Always deployable
-  Changes validated by CI
-  Affects everyone

##  **The Safe State Guarantee**

- **Main branch** = Always stable, tested, production-ready
- **Feature branches** = Your experimental workspace
- **CI Pipeline** = Automatically validates every change
- **Pull Requests** = Review process before merging

##  **Module Development Examples**

`ash
# Work on specific modules
git checkout -b feature/improve-vector-store
# Edit: src/lumina_memory/vector_store.py
# Test: tests/test_vector_store.py

git checkout -b feature/add-quantum-memory  
# Create: src/lumina_memory/quantum.py
# Test: tests/test_quantum.py

git checkout -b feature/optimize-embeddings
# Edit: src/lumina_memory/embeddings.py
# Benchmark: scripts/profile_retrieval.py
`

##  **Useful Commands**

`ash
# Check repository status
git status
git log --oneline -5

# See what changed  
git diff
git diff --cached

# Undo changes (careful!)
git restore filename.py      # Undo unstaged changes
git restore --staged file.py # Unstage changes
git reset --soft HEAD~1      # Undo last commit (keep changes)

# Branch management
git branch -a                # List all branches
git checkout main           # Switch to main
git branch -d feature-name  # Delete merged branch

# Remote updates
git fetch origin            # Get latest from GitHub
git pull origin main        # Update main branch
`

##  **Important Rules**

1. **Never work directly on main** - Always use feature branches
2. **Commit often** - Small, focused commits are better
3. **Write descriptive messages** - Future you will thank you
4. **Test before pushing** - Run tests locally when possible
5. **Pull before pushing** - Always get latest changes first

##  **Your Repository Structure**

`
lumina-memory-system/
 src/lumina_memory/        # Core modules
    memory_system.py      # Main memory system
    embeddings.py         # Embedding providers  
    vector_store.py       # Vector storage
    analytics.py          # Performance analytics
    ...
 tests/                    # Comprehensive test suite
 scripts/                  # Utilities (performance profiling)
 data/sample/             # Sample datasets for testing
 .github/workflows/       # CI/CD automation
 notebooks/              # Jupyter notebooks for demos
`

##  **Need Help?**

- Check the CI logs if builds fail
- Look at existing code for patterns
- Each module has tests showing usage examples
- Performance profiling available via scripts/profile_retrieval.py

**Happy coding! **
