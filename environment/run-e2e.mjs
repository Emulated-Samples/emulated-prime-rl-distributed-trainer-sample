/**
 * End-to-End Test Runner
 * 
 * Runs setupProblem followed by runTests to validate the full environment.
 */

import env from './dist/index.js';

const logger = {
  info: (...args) => console.log('[INFO]', new Date().toISOString(), ...args),
  warn: (...args) => console.log('[WARN]', new Date().toISOString(), ...args),
  error: (...args) => console.error('[ERROR]', new Date().toISOString(), ...args),
};

async function main() {
  console.log('========================================');
  console.log('  Prime-RL End-to-End Test');
  console.log('========================================');
  console.log(`Started: ${new Date().toISOString()}`);
  console.log('');

  try {
    // Step 1: Setup
    console.log('>>> Step 1: Running setupProblem()...');
    console.log('    This creates EKS cluster, GPU operator, EFS, and deploys prime-rl');
    console.log('    Expected time: ~30-40 minutes');
    console.log('');
    
    await env.setupProblem('deploy-prime-rl', logger);
    
    console.log('');
    console.log('>>> Step 1 Complete: All resources created');
    console.log('');

    // Step 2: Verify
    console.log('>>> Step 2: Running runTests() to verify...');
    console.log('');
    
    const results = await env.runTests('deploy-prime-rl', logger);
    
    console.log('');
    console.log('========================================');
    console.log('  Test Results');
    console.log('========================================');
    
    let passed = 0;
    let failed = 0;
    
    for (const result of results) {
      const status = result.status === 'passed' ? '✓ PASS' : '✗ FAIL';
      console.log(`${status}: ${result.name || result.id}`);
      if (result.status !== 'passed' && result.error) {
        console.log(`       ${result.error}`);
      }
      result.status === 'passed' ? passed++ : failed++;
    }
    
    console.log('');
    console.log(`Total: ${passed} passed, ${failed} failed`);
    console.log(`Finished: ${new Date().toISOString()}`);
    
    if (failed > 0) {
      process.exit(1);
    }
    
  } catch (error) {
    console.error('');
    console.error('>>> FATAL ERROR:');
    console.error(error);
    process.exit(1);
  }
}

main();
