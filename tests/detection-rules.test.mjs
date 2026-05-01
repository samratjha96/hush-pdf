import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const html = fs.readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
const start = html.indexOf('function _looksLikePrivateSpan');
const end = html.indexOf('function _newSessionId');

assert.notEqual(start, -1, 'rule helper block start not found');
assert.notEqual(end, -1, 'rule helper block end not found');

const sandbox = {};
vm.runInNewContext(
  html.slice(start, end) + '\nthis.__rules = { _findRuleSpans, _luhnOk };',
  sandbox
);

const { _findRuleSpans, _luhnOk } = sandbox.__rules;

const text = [
  'Email ada@example.com and call (415) 555-1212.',
  'SSN 123-45-6789.',
  'Card 4111 1111 1111 1111.',
  'Policy number ABC-12345.',
  'api_key=sk_test_1234567890abcdef',
  'Visit https://example.com/private.',
].join('\n');

const labels = _findRuleSpans(text).map(span => span.label);

assert(labels.includes('PRIVATE_EMAIL'), 'email should be detected');
assert(labels.includes('PRIVATE_PHONE'), 'phone should be detected');
assert(labels.filter(label => label === 'ACCOUNT_NUMBER').length >= 3, 'SSN, card, and policy IDs should be detected');
assert(labels.includes('SECRET'), 'secret should be detected');
assert(labels.includes('PRIVATE_URL'), 'URL should be detected');
assert.equal(_luhnOk('4111 1111 1111 1111'), true, 'known test card should pass Luhn');
assert.equal(_luhnOk('4111 1111 1111 1112'), false, 'invalid card should fail Luhn');

console.log('detection rules ok');
