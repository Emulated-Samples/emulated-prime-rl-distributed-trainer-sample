/**
 * Cleanup Script
 * 
 * Runs cleanup to delete all resources.
 */

import env from './dist/index.js';

const logger = {
  info: (...args) => console.log('[INFO]', new Date().toISOString(), ...args),
  warn: (...args) => console.log('[WARN]', new Date().toISOString(), ...args),
  error: (...args) => console.error('[ERROR]', new Date().toISOString(), ...args),
};

try {
  console.log('Running cleanup...');
  await env.cleanup(logger);
  console.log('Cleanup completed successfully!');
} catch (error) {
  console.error('Cleanup failed:', error);
  process.exit(1);
}
