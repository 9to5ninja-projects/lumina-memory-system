"""
Class Registry - Live System Architecture Management
Provides runtime access to class analysis and conflict detection.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

class ConflictType(Enum):
    CONFIGURATION = "configuration"
    MEMORY_CLASS = "memory_class"
    KERNEL_PROLIFERATION = "kernel_proliferation"
    NAMING_COLLISION = "naming_collision"
    IMPLEMENTATION_DUPLICATE = "implementation_duplicate"

@dataclass
class ClassInfo:
    """Information about a class in the system"""
    name: str
    location: str  # "xp_core", "bridge", "hd_kernel", "main_branch"  
    line_number: Optional[int]
    status: str  # "DEFINED", "STUB", "REFERENCED", "CONFLICT"
    conflicts_with: List[str]
    description: str

class ClassRegistry:
    """
    Central registry of all classes across the lumina memory system.
    Provides conflict detection and architectural guidance.
    """
    
    def __init__(self):
        self.classes: Dict[str, List[ClassInfo]] = {}
        self.conflicts: Dict[ConflictType, List[str]] = {}
        self._load_known_classes()
    
    def _load_known_classes(self):
        """Load known class information from our comprehensive analysis"""
        
        # XP Core Classes (7 total)
        xp_core_classes = [
            ClassInfo("HybridLexicalAttributor", "xp_core", 1304, "DEFINED", [], "Lexical attribution with decay mathematics"),
            ClassInfo("HolographicShapeComputer", "xp_core", 1366, "DEFINED", [], "Vector shape computation system"), 
            ClassInfo("FastLexicalAttributorDemo", "xp_core", 1558, "DEFINED", [], "Demo lexical attribution system"),
            ClassInfo("MemoryUnit", "xp_core", 1121, "DEFINED", ["bridge.Memory", "main_branch.Memory"], "Complete 13-component holographic implementation"),
            ClassInfo("MemoryUnit", "xp_core", 1742, "DEFINED", ["bridge.Memory", "main_branch.Memory"], "Versioned store framework implementation"),
            ClassInfo("VersionedXPStore", "xp_core", 1700, "COMPLETE", ["main_branch.versioned_xp_store"], "CRYPTOGRAPHIC VERSIONING SYSTEM: Full Git-like branching with SHA-256 commit integrity, cryptographic memory unit identity tracking, temporal provenance with mathematical immutability guarantees"),
            ClassInfo("SpacyLexicalAttributor", "xp_core", 1500, "DEFINED", [], "SpaCy-based lexical attribution"),
        ]
        
        # Bridge Classes (Key ones from 18 total)
        bridge_classes = [
            ClassInfo("XPCoreConfig", "bridge", 320, "DEFINED", ["xp_core.XPCoreConfig", "main_branch.UnifiedConfig"], "Configuration conflict - multiple definitions"),
            ClassInfo("XPCoreBridge", "bridge", 327, "DEFINED", [], "Main integration bridge between XP Core and Unit-Space"),
            ClassInfo("SpaceConfig", "bridge", 606, "DEFINED", ["main_branch.UnifiedConfig"], "Unit-space specific settings"),
            ClassInfo("SpaceManager", "bridge", 619, "DEFINED", [], "KNN topology management - Bridge exclusive"),
            ClassInfo("UnitSpaceKernel", "bridge", 1024, "DEFINED", ["main_branch.UnifiedKernel", "hd_kernel.XPKernel"], "Main kernel implementation - proliferation conflict"),
            ClassInfo("Memory", "bridge", 1777, "DEFINED", ["xp_core.MemoryUnit", "main_branch.Memory"], "Spatial memory representation - trinity conflict"),
            ClassInfo("UnifiedMemory", "bridge", 2463, "DEFINED", ["main_branch.UnifiedMemory"], "Duplicate unified memory definitions"),
        ]
        
        # HD Kernel Classes (2 total) 
        hd_kernel_classes = [
            ClassInfo("XPKernel", "hd_kernel", 89, "DEFINED", [], "Abstract base class specification - interface contract"),
            ClassInfo("MyCustomKernel", "hd_kernel", 161, "DEFINED", [], "Example implementation showing integration pattern"),
        ]
        
        # Main Branch Classes (Key unified approach)
        main_branch_classes = [
            ClassInfo("UnifiedMemory", "main_branch", 22, "DEFINED", ["bridge.UnifiedMemory"], "Unified memory representation"),
            ClassInfo("UnifiedConfig", "main_branch", 59, "DEFINED", ["xp_core.XPCoreConfig", "bridge.SpaceConfig"], "Unified configuration approach"),
            ClassInfo("UnifiedKernel", "main_branch", 86, "DEFINED", ["bridge.UnitSpaceKernel"], "Unified kernel supporting all patterns"),
        ]
        
        # Register all classes
        all_classes = xp_core_classes + bridge_classes + hd_kernel_classes + main_branch_classes
        for class_info in all_classes:
            if class_info.name not in self.classes:
                self.classes[class_info.name] = []
            self.classes[class_info.name].append(class_info)
        
        # Register known conflicts
        self.conflicts = {
            ConflictType.CONFIGURATION: ["XPCoreConfig", "SpaceConfig", "UnifiedConfig"],
            ConflictType.MEMORY_CLASS: ["Memory", "MemoryUnit", "UnifiedMemory"], 
            ConflictType.KERNEL_PROLIFERATION: ["Kernel", "UnitSpaceKernel", "UnifiedKernel", "XPKernel"],
            ConflictType.IMPLEMENTATION_DUPLICATE: ["UnifiedMemory", "XPCoreConfig"],
        }
    
    def check_conflicts(self, class_name: str) -> List[ConflictType]:
        """Check if a class name has known conflicts"""
        conflicts = []
        for conflict_type, class_list in self.conflicts.items():
            if class_name in class_list:
                conflicts.append(conflict_type)
        return conflicts
    
    def get_class_info(self, class_name: str) -> List[ClassInfo]:
        """Get all instances of a class across the system"""
        return self.classes.get(class_name, [])
    
    def suggest_unified_approach(self, class_name: str) -> Optional[str]:
        """Suggest unified approach for conflicted classes"""
        suggestions = {
            "XPCoreConfig": "Use UnifiedConfig from main branch - consolidates all configuration approaches",
            "SpaceConfig": "Use UnifiedConfig from main branch - consolidates all configuration approaches", 
            "Memory": "Use UnifiedMemory from main branch - supports all memory patterns",
            "MemoryUnit": "Use UnifiedMemory from main branch - supports all memory patterns",
            "UnitSpaceKernel": "Use UnifiedKernel from main branch - implements XPKernel interface",
        }
        return suggestions.get(class_name)
    
    def print_system_status(self):
        """Print current system architectural status"""
        print("🔬 LUMINA MEMORY SYSTEM - ARCHITECTURAL STATUS")
        print("=" * 50)
        
        locations = {"xp_core": 0, "bridge": 0, "hd_kernel": 0, "main_branch": 0}
        conflict_count = 0
        
        for class_name, instances in self.classes.items():
            for instance in instances:
                locations[instance.location] += 1
                if instance.conflicts_with:
                    conflict_count += 1
        
        print(f"📊 Class Distribution:")
        for location, count in locations.items():
            print(f"   {location}: {count} classes")
        
        print(f"\n🚨 Conflicts: {conflict_count} classes have conflicts")
        print(f"🎯 Major Conflict Types: {len(self.conflicts)} identified patterns")
        
        print(f"\n✅ Unified Foundation Status:")
        unified_classes = [cls for instances in self.classes.values() for cls in instances if cls.location == "main_branch"]
        print(f"   Unified classes available: {len(unified_classes)}")
        
    def get_maintenance_guidance(self) -> str:
        """Get guidance for maintaining class architecture"""
        return """
🔄 CLASS MAINTENANCE PROTOCOL:

BEFORE adding new classes:
1. Check class_registry.check_conflicts(class_name) 
2. Review docs/CLASS_ANALYSIS.md for architectural context
3. Consider using existing unified classes instead

AFTER adding new classes:
1. Update docs/CLASS_ANALYSIS.md with new class information
2. Update class_registry.py with new class entries
3. Cross-reference with other notebooks for conflicts
4. Test unified foundation integration

GOAL: All classes work through unified foundation following HD Kernel interface
        """

# Global registry instance
registry = ClassRegistry()

def check_class_conflict(class_name: str) -> bool:
    """Quick conflict check for use in notebooks"""
    conflicts = registry.check_conflicts(class_name)
    if conflicts:
        print(f"⚠️ WARNING: '{class_name}' has conflicts: {[c.value for c in conflicts]}")
        suggestion = registry.suggest_unified_approach(class_name)
        if suggestion:
            print(f"💡 SUGGESTION: {suggestion}")
        return True
    return False

def print_class_info(class_name: str):
    """Print comprehensive class information"""
    instances = registry.get_class_info(class_name)
    if not instances:
        print(f"❌ Class '{class_name}' not found in registry")
        return
    
    print(f"🔍 CLASS INFO: {class_name}")
    print("-" * 30)
    for i, instance in enumerate(instances, 1):
        print(f"{i}. Location: {instance.location}")
        print(f"   Line: {instance.line_number}")
        print(f"   Status: {instance.status}")  
        print(f"   Description: {instance.description}")
        if instance.conflicts_with:
            print(f"   ⚠️ Conflicts: {instance.conflicts_with}")
        print()

if __name__ == "__main__":
    registry.print_system_status()
    print(registry.get_maintenance_guidance())
