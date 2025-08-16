# Class Analysis Across Notebooks

## XP Core Design Notebook

### Implementation Classes (with __init__ methods):
- SpacyLexicalAttributor (line ~291)
- HybridLexicalAttributor (line ~471) 
- VersionedXPStore (line ~719)
- HolographicShapeComputer (line ~1163)
- Unknown class (line ~1561)
- Unknown class (line ~1725)

### Class Usage/References Found:
- MemoryUnit (constructor usage found, definition TBD)
- HolographicShapeComputer (instantiated as shape_computer)

## Unit Space Kernel Bridge Notebook

### Currently Executed Classes:
- XPCoreBridge (active in kernel)

### Need to check for class definitions and conflicts

## HD Kernel XP Spec Notebook

### Status: Not yet analyzed (7 markdown cells)

## Analysis Plan
1. Find actual implementation code for each class with __init__
2. Check for naming conflicts between notebooks  
3. Identify stubs vs complete implementations
4. Map dependencies and relationships
