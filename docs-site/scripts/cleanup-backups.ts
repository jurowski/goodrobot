import fs from 'fs';
import path from 'path';
import cron from 'node-cron';

interface BackupCleanupOptions {
  maxAgeDays?: number;
  maxCount?: number;
  dryRun?: boolean;
  verbose?: boolean;
  schedule?: string;
  validateContent?: boolean;
  minSizeKB?: number;
  maxSizeKB?: number;
}

interface BackupFile {
  filename: string;
  timestamp: Date;
  ageInDays: number;
  sizeKB: number;
  isValid: boolean;
}

interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

export function validateBackupFile(filePath: string): ValidationResult {
  const result: ValidationResult = {
    isValid: true,
    errors: []
  };

  try {
    // Check if file exists and is readable
    if (!fs.existsSync(filePath)) {
      result.isValid = false;
      result.errors.push('File does not exist');
      return result;
    }

    // Check file size
    const stats = fs.statSync(filePath);
    if (stats.size === 0) {
      result.isValid = false;
      result.errors.push('File is empty');
    }

    // Check if file is valid JSON
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      JSON.parse(content);
    } catch (error) {
      result.isValid = false;
      result.errors.push('Invalid JSON format');
    }

    return result;
  } catch (error) {
    result.isValid = false;
    result.errors.push(`Error validating file: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return result;
  }
}

function getBackupAge(filename: string): number {
  const timestampMatch = filename.match(/\d{14}/);
  if (!timestampMatch) return Infinity;
  
  const timestamp = timestampMatch[0];
  const backupDate = new Date(
    parseInt(timestamp.substring(0, 4)), // year
    parseInt(timestamp.substring(4, 6)) - 1, // month (0-indexed)
    parseInt(timestamp.substring(6, 8)), // day
    parseInt(timestamp.substring(8, 10)), // hour
    parseInt(timestamp.substring(10, 12)), // minute
    parseInt(timestamp.substring(12, 14)) // second
  );
  
  const now = new Date();
  return (now.getTime() - backupDate.getTime()) / (1000 * 60 * 60 * 24);
}

export function getBackupFiles(backupDir: string, options: BackupCleanupOptions): BackupFile[] {
  if (!fs.existsSync(backupDir)) {
    return [];
  }

  return fs.readdirSync(backupDir)
    .filter(file => file.endsWith('.json'))
    .map(filename => {
      const filePath = path.join(backupDir, filename);
      const stats = fs.statSync(filePath);
      const validation = options.validateContent ? validateBackupFile(filePath) : { isValid: true, errors: [] };
      
      return {
        filename,
        timestamp: new Date(getBackupAge(filename)),
        ageInDays: getBackupAge(filename),
        sizeKB: stats.size / 1024,
        isValid: validation.isValid
      };
    })
    .filter(file => {
      if (options.minSizeKB && file.sizeKB < options.minSizeKB) return false;
      if (options.maxSizeKB && file.sizeKB > options.maxSizeKB) return false;
      if (options.validateContent && !file.isValid) return false;
      return true;
    })
    .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

export function cleanupBackups(backupDir: string, options: BackupCleanupOptions = {}): void {
  const {
    maxAgeDays = 30,
    maxCount = 10,
    dryRun = false,
    verbose = false,
    validateContent = false,
    minSizeKB,
    maxSizeKB
  } = options;

  const backupFiles = getBackupFiles(backupDir, options);
  const filesToDelete: BackupFile[] = [];

  // Find files older than maxAgeDays
  if (maxAgeDays) {
    filesToDelete.push(...backupFiles.filter(file => file.ageInDays > maxAgeDays));
  }

  // Find files beyond maxCount
  if (maxCount && backupFiles.length > maxCount) {
    filesToDelete.push(...backupFiles.slice(maxCount));
  }

  // Remove duplicates
  const uniqueFilesToDelete = Array.from(new Set(filesToDelete.map(f => f.filename)))
    .map(filename => filesToDelete.find(f => f.filename === filename)!);

  if (verbose) {
    console.log('Backup cleanup report:');
    console.log(`Total backups found: ${backupFiles.length}`);
    console.log(`Files to delete: ${uniqueFilesToDelete.length}`);
    console.log('Files to be deleted:');
    uniqueFilesToDelete.forEach(file => {
      console.log(`- ${file.filename} (${file.ageInDays.toFixed(1)} days old, ${file.sizeKB.toFixed(2)} KB)`);
      if (!file.isValid) {
        console.log('  Invalid backup file');
      }
    });
  }

  if (dryRun) {
    console.log('Dry run: No files will be deleted');
    return;
  }

  // Delete files
  uniqueFilesToDelete.forEach(file => {
    try {
      fs.unlinkSync(path.join(backupDir, file.filename));
      if (verbose) {
        console.log(`Deleted: ${file.filename}`);
      }
    } catch (error) {
      console.error(`Error deleting ${file.filename}:`, error instanceof Error ? error.message : 'Unknown error');
    }
  });

  if (verbose) {
    console.log('Cleanup completed');
  }
}

function scheduleCleanup(backupDir: string, options: BackupCleanupOptions): void {
  if (!options.schedule) {
    console.error('No schedule specified');
    return;
  }

  if (!cron.validate(options.schedule)) {
    console.error('Invalid cron schedule');
    return;
  }

  console.log(`Scheduling cleanup with cron: ${options.schedule}`);
  cron.schedule(options.schedule, () => {
    console.log('Running scheduled cleanup...');
    cleanupBackups(backupDir, options);
  });
}

function main() {
  const args = process.argv.slice(2);
  const backupDir = path.join(__dirname, '../backups');
  
  const options: BackupCleanupOptions = {
    maxAgeDays: 30,
    maxCount: 10,
    dryRun: false,
    verbose: false,
    validateContent: false
  };

  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--max-age':
        options.maxAgeDays = parseInt(args[++i]);
        break;
      case '--max-count':
        options.maxCount = parseInt(args[++i]);
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--verbose':
        options.verbose = true;
        break;
      case '--schedule':
        options.schedule = args[++i];
        break;
      case '--validate':
        options.validateContent = true;
        break;
      case '--min-size':
        options.minSizeKB = parseInt(args[++i]);
        break;
      case '--max-size':
        options.maxSizeKB = parseInt(args[++i]);
        break;
      case '--help':
        printHelp();
        return;
      default:
        console.error(`Unknown option: ${arg}`);
        printHelp();
        return;
    }
  }

  if (options.schedule) {
    scheduleCleanup(backupDir, options);
  } else {
    cleanupBackups(backupDir, options);
  }
}

function printHelp() {
  console.log(`
Backup Cleanup Utility

Usage:
  ts-node cleanup-backups.ts [options]

Options:
  --max-age DAYS     Maximum age of backups in days (default: 30)
  --max-count COUNT  Maximum number of backups to keep (default: 10)
  --dry-run          Show what would be deleted without actually deleting
  --verbose          Show detailed information about the cleanup process
  --schedule CRON    Schedule cleanup using cron expression
  --validate         Validate backup file content
  --min-size KB      Minimum file size in KB
  --max-size KB      Maximum file size in KB
  --help             Show this help message

Examples:
  # Clean up backups older than 30 days, keeping at most 10 backups
  ts-node cleanup-backups.ts

  # Clean up backups older than 7 days, keeping at most 5 backups
  ts-node cleanup-backups.ts --max-age 7 --max-count 5

  # Show what would be deleted without actually deleting
  ts-node cleanup-backups.ts --dry-run --verbose

  # Schedule daily cleanup at midnight
  ts-node cleanup-backups.ts --schedule "0 0 * * *"

  # Validate backup files and enforce size limits
  ts-node cleanup-backups.ts --validate --min-size 1 --max-size 1000
  `);
}

if (require.main === module) {
  main();
} 