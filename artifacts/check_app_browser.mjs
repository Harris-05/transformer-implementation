import {Browser,sleep} from 'file:///C:/Users/EliteSeries/.codex/skills/taste-skill/scripts/lib/cdp.mjs';
import {mkdirSync,writeFileSync} from 'node:fs';
const browser=await Browser.launch();
const findings=[];
try{
 for(const width of [1440,768,390]){
  const page=await browser.newPage({width,height:1000,reducedMotion:true});
  await page.goto('http://127.0.0.1:8765');
  await page.eval(async()=>{await document.fonts.ready;return true;});
  async function click(selector){const doc=await page.send('DOM.getDocument');const selected=await page.send('DOM.querySelector',{nodeId:doc.root.nodeId,selector});await page.send('DOM.scrollIntoViewIfNeeded',{nodeId:selected.nodeId});const rect=await page.eval(s=>{const r=document.querySelector(s).getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2};},selector);await page.send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...rect});await page.send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...rect});await sleep(150);}
  await click('#run-screen');
  let value=await page.eval(()=>({error:document.getElementById('message').textContent,overflow:document.documentElement.scrollWidth>innerWidth,fonts:[document.fonts.check('15px "Instrument Sans"'),document.fonts.check('32px "Instrument Serif"')]}));
  if(!value.error.includes('patient record')||value.overflow||value.fonts.some(x=>!x))throw new Error('Invalid empty state / overflow / fonts: '+JSON.stringify(value));
  await click('#example');
  const count=await page.eval(()=>document.getElementById('input-count').textContent);
  if(!count.includes('3 criteria'))throw new Error('Example criteria count');
  await page.screenshot({path:`artifacts/app_browser_final/synthetic-${width}.png`,format:'png',fullPage:true});
  await click('[data-tab="benchmark"]');
  const lab=await page.eval(()=>({visible:!document.getElementById('benchmark').hidden,title:document.getElementById('page-title').textContent}));
  if(!lab.visible)throw new Error('Tab did not change');
  findings.push({width,...value,exampleCriteria:count,lab:lab.visible,consoleErrors:page.consoleErrors});
  await page.close();
 }
 writeFileSync('artifacts/app_browser_final/workflow_checks.json',JSON.stringify(findings,null,2));
 console.log('PASS: real browser at 1440/768/390, font readiness, invalid input, synthetic fill, tabs, overflow and console checks.');
 console.log(JSON.stringify(findings));
}finally{await browser.close();}
