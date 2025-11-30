#!/usr/bin/env python3
"""
Test musicnn Library - Quick library test
Tests pre-trained musicnn models (MTT and MSD)
Requires: pip install musicnn
"""

from musicnn.tagger import top_tags

tags = top_tags('./audio/example.mp3', model='MTT_musicnn', topN=5)
print("MTT_musicnn tags:", tags)

tags = top_tags('./audio/example.mp3', model='MSD_musicnn', topN=5)
print("MSD_musicnn tags:", tags)
