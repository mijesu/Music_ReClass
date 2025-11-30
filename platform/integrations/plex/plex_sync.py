#!/usr/bin/env python3
"""Plex Library Sync - Read 100k songs from Plex and store in database"""

from plexapi.server import PlexServer
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

# Configuration
PLEX_URL = 'http://localhost:32400'
PLEX_TOKEN = 'YOUR_PLEX_TOKEN_HERE'  # TODO: Get from Plex Web
PLEX_LIBRARY_NAME = 'Music'
DB_PATH = 'music_100k.db'

class PlexSync:
    def __init__(self):
        self.plex = PlexServer(PLEX_URL, PLEX_TOKEN)
        self.db = sqlite3.connect(DB_PATH)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables if not exist"""
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS songs (
                song_id INTEGER PRIMARY KEY AUTOINCREMENT,
                plex_id TEXT UNIQUE NOT NULL,
                file_path TEXT NOT NULL,
                file_name TEXT,
                
                title TEXT,
                artist TEXT,
                album TEXT,
                year INTEGER,
                track_number INTEGER,
                duration REAL,
                genre_tag TEXT,
                
                added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_synced TIMESTAMP,
                is_analyzed BOOLEAN DEFAULT 0
            )
        ''')
        self.db.commit()
    
    def get_music_library(self):
        """Get Plex music library"""
        try:
            library = self.plex.library.section(PLEX_LIBRARY_NAME)
            return library
        except Exception as e:
            print(f"Error: Could not find library '{PLEX_LIBRARY_NAME}'")
            print(f"Available libraries: {[s.title for s in self.plex.library.sections()]}")
            sys.exit(1)
    
    def sync_library(self):
        """Sync all tracks from Plex to database"""
        library = self.get_music_library()
        
        print(f"Syncing Plex library: {PLEX_LIBRARY_NAME}")
        print(f"Total tracks in Plex: {library.totalSize}")
        
        cursor = self.db.cursor()
        synced = 0
        errors = 0
        
        for track in library.all():
            try:
                # Get file path
                file_path = track.media[0].parts[0].file if track.media else None
                
                if not file_path:
                    print(f"Warning: No file path for {track.title}")
                    errors += 1
                    continue
                
                # Insert or update
                cursor.execute('''
                    INSERT OR REPLACE INTO songs 
                    (plex_id, file_path, file_name, title, artist, album, 
                     year, track_number, duration, genre_tag, last_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(track.ratingKey),
                    file_path,
                    Path(file_path).name,
                    track.title,
                    track.artist().title if track.artist() else None,
                    track.album().title if track.album() else None,
                    track.year,
                    track.trackNumber,
                    track.duration / 1000.0 if track.duration else None,
                    ', '.join([g.tag for g in track.genres]) if track.genres else None,
                    datetime.now()
                ))
                
                synced += 1
                if synced % 1000 == 0:
                    print(f"Synced {synced} tracks...")
                    self.db.commit()
                
            except Exception as e:
                print(f"Error syncing track: {e}")
                errors += 1
                continue
        
        self.db.commit()
        print(f"\n✅ Sync complete!")
        print(f"   Synced: {synced} tracks")
        print(f"   Errors: {errors} tracks")
        
        return synced, errors
    
    def get_stats(self):
        """Get database statistics"""
        cursor = self.db.cursor()
        
        total = cursor.execute('SELECT COUNT(*) FROM songs').fetchone()[0]
        analyzed = cursor.execute('SELECT COUNT(*) FROM songs WHERE is_analyzed = 1').fetchone()[0]
        
        print(f"\nDatabase Statistics:")
        print(f"  Total songs: {total:,}")
        print(f"  Analyzed: {analyzed:,}")
        print(f"  Pending: {(total - analyzed):,}")
    
    def test_file_access(self, limit=10):
        """Test if files are accessible"""
        cursor = self.db.cursor()
        cursor.execute('SELECT file_path FROM songs LIMIT ?', (limit,))
        
        print(f"\nTesting file access (first {limit} files):")
        accessible = 0
        
        for row in cursor.fetchall():
            file_path = row[0]
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
                accessible += 1
            else:
                print(f"  ❌ {file_path}")
        
        print(f"\nAccessible: {accessible}/{limit}")
        
        if accessible == 0:
            print("\n⚠️  Warning: No files accessible!")
            print("   Check if Plex library path matches local filesystem")

if __name__ == '__main__':
    print("=== Plex Library Sync ===\n")
    
    # Check token
    if PLEX_TOKEN == 'YOUR_PLEX_TOKEN_HERE':
        print("❌ Error: Please set your Plex token in the script")
        print("\nHow to get token:")
        print("1. Open Plex Web: http://localhost:32400/web")
        print("2. Play any song")
        print("3. Click ⋮ → Get Info → View XML")
        print("4. Copy X-Plex-Token from URL")
        sys.exit(1)
    
    try:
        sync = PlexSync()
        
        # Sync library
        sync.sync_library()
        
        # Show stats
        sync.get_stats()
        
        # Test file access
        sync.test_file_access()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Is Plex running? Check http://localhost:32400/web")
        print("- Is token correct?")
        print("- Is library name correct?")
