import fs from 'fs';
import path from 'path';
import { DocumentationSystem } from '../src/services/documentation';

interface SearchOptions {
  dryRun?: boolean;
  verbose?: boolean;
  databases?: string[];
  backup?: boolean;
  restore?: string;
}

function listBackups(): string[] {
  const backupDir = path.join(__dirname, '../backups');
  if (!fs.existsSync(backupDir)) {
    return [];
  }
  return fs.readdirSync(backupDir)
    .filter(file => file.startsWith('documentation-') && file.endsWith('.json'))
    .sort()
    .reverse();
}

function restoreBackup(backupFile: string): void {
  const backupDir = path.join(__dirname, '../backups');
  const backupPath = path.join(backupDir, backupFile);
  const docPath = path.join(__dirname, '../public/data/documentation.json');
  
  if (!fs.existsSync(backupPath)) {
    throw new Error(`Backup file ${backupFile} not found`);
  }
  
  fs.copyFileSync(backupPath, docPath);
  console.log(`Successfully restored from backup: ${backupFile}`);
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  
  // Handle restore command
  if (args[0] === '--restore') {
    const backups = listBackups();
    if (backups.length === 0) {
      console.error('No backups found');
      process.exit(1);
    }
    
    if (args[1]) {
      restoreBackup(args[1]);
    } else {
      console.log('Available backups:');
      backups.forEach((backup, index) => {
        console.log(`${index + 1}. ${backup}`);
      });
    }
    return;
  }
  
  if (args.length < 2) {
    console.error('Usage:');
    console.error('  Update documentation:');
    console.error('    ts-node update-documentation-enhanced.ts <system-id> <keyword1> [keyword2 ...] [options]');
    console.error('  Restore from backup:');
    console.error('    ts-node update-documentation-enhanced.ts --restore [backup-file]');
    console.error('Options:');
    console.error('  --dry-run     Preview changes without saving');
    console.error('  --verbose     Show detailed output');
    console.error('  --databases   Comma-separated list of databases (semantic-scholar,arxiv,ieee,acm)');
    console.error('  --backup      Create backup before making changes');
    process.exit(1);
  }

  // ... rest of the function ...
} 