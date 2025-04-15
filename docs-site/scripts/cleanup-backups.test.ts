import fs from 'fs';
import path from 'path';
import { jest } from '@jest/globals';
import { cleanupBackups, getBackupFiles, validateBackupFile } from './cleanup-backups';

// Mock fs module
jest.mock('fs');

describe('Backup Cleanup Tests', () => {
  const mockBackupDir = '/mock/backup/dir';
  const mockOptions = {
    maxAgeDays: 30,
    maxCount: 10,
    dryRun: true,
    verbose: true
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('validateBackupFile', () => {
    it('should validate a valid backup file', () => {
      const mockFilePath = '/mock/backup/file.json';
      const mockContent = JSON.stringify({ test: 'data' });
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });
      (fs.readFileSync as jest.Mock).mockReturnValue(mockContent);

      const result = validateBackupFile(mockFilePath);
      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect invalid JSON', () => {
      const mockFilePath = '/mock/backup/invalid.json';
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });
      (fs.readFileSync as jest.Mock).mockReturnValue('{invalid json}');

      const result = validateBackupFile(mockFilePath);
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('Invalid JSON format');
    });

    it('should detect empty file', () => {
      const mockFilePath = '/mock/backup/empty.json';
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 0 });

      const result = validateBackupFile(mockFilePath);
      expect(result.isValid).toBe(false);
      expect(result.errors).toContain('File is empty');
    });
  });

  describe('getBackupFiles', () => {
    it('should return sorted backup files', () => {
      const mockFiles = [
        'backup_20230101120000.json',
        'backup_20230102120000.json',
        'backup_20230103120000.json'
      ];
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(mockFiles);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });

      const files = getBackupFiles(mockBackupDir, mockOptions);
      expect(files).toHaveLength(3);
      expect(files[0].filename).toBe('backup_20230103120000.json');
    });

    it('should filter files by size', () => {
      const mockFiles = ['backup_20230101120000.json'];
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(mockFiles);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 50 * 1024 }); // 50KB

      const options = { ...mockOptions, minSizeKB: 100 };
      const files = getBackupFiles(mockBackupDir, options);
      expect(files).toHaveLength(0);
    });
  });

  describe('cleanupBackups', () => {
    it('should delete old backups', () => {
      const mockFiles = [
        'backup_20220101120000.json', // 1 year old
        'backup_20221201120000.json', // 1 month old
        'backup_20230101120000.json'  // current
      ];
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(mockFiles);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });
      (fs.unlinkSync as jest.Mock).mockImplementation(() => {});

      cleanupBackups(mockBackupDir, { ...mockOptions, maxAgeDays: 60 });
      expect(fs.unlinkSync).toHaveBeenCalledTimes(1);
    });

    it('should respect maxCount', () => {
      const mockFiles = Array(15).fill(0).map((_, i) => 
        `backup_202301${String(i + 1).padStart(2, '0')}120000.json`
      );
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(mockFiles);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });
      (fs.unlinkSync as jest.Mock).mockImplementation(() => {});

      cleanupBackups(mockBackupDir, { ...mockOptions, maxCount: 10 });
      expect(fs.unlinkSync).toHaveBeenCalledTimes(5);
    });

    it('should not delete files in dry run mode', () => {
      const mockFiles = ['backup_20220101120000.json'];
      
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(mockFiles);
      (fs.statSync as jest.Mock).mockReturnValue({ size: 100 });

      cleanupBackups(mockBackupDir, { ...mockOptions, dryRun: true });
      expect(fs.unlinkSync).not.toHaveBeenCalled();
    });
  });
}); 