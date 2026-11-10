"""Generate IEEE-paper tables/figures from saved experiments, never invented scores."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent


def tex(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(value))


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def predictions(path):return [json.loads(line) for line in (ROOT/path).read_text(encoding='utf-8').splitlines()]


def table(path, header, rows, columns):
    lines=[r'\begin{tabular}{'+columns+'}',r'\toprule', ' & '.join(header)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--final-output',type=Path,default=ROOT/'results/final_test');args=parser.parse_args()
    out=ROOT/'report/generated';out.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    before=read('results/baseline_validation/metrics.json');after=read('results/finetuning_qlora_v1/validation/metrics.json')
    kv=read('results/kv_cache_math_v1/metrics.json');attention=read('results/attention_gpu/metrics.json');spec=read('results/speculative_v1/metrics.json')
    rows=[]
    for group in ('overall','inclusion','exclusion'):
        for name,metrics in [('Original',before),('Adapted',after)]:
            g=metrics['groups'][group];rows.append([group,name,g['records'],f"{g['classification']['accuracy']:.4f}",f"{g['classification']['macro_f1']:.4f}",f"{g['evidence']['micro']['f1']:.4f}",f"{g['output_valid_rate']:.4f}"])
    table(out/'validation_table.tex',['Group','Model','N','Accuracy','Macro F1','Evidence F1','Valid'],rows,'llrrrrr')
    per_class=after['groups']['overall']['classification']['per_class']
    rows=[]
    for label,values in per_class.items():
        if label=='__invalid__':continue
        rows.append([tex(label),int(values['support']),f"{values['precision']:.3f}",f"{values['recall']:.3f}",f"{values['f1-score']:.3f}"])
    table(out/'class_table.tex',['Category','N','Precision','Recall','F1'],rows,'lrrrr')
    rows=[]
    for case in kv['cases']:
        modes=case['modes'];speed=case['total_speedup_uncached_over_cached']
        rows.append([tex(case['case_id']),f"{modes['cached']['total_seconds_median']:.3f}",f"{modes['uncached']['total_seconds_median']:.3f}",f'{speed:.2f}' if speed is not None else 'unequal'])
    table(out/'kv_table.tex',['Input/output case','Cached (s)','Uncached (s)','Speedup'],rows,'lrrr')
    rows=[]
    for c in attention['cases']:
        m=c['timing']['methods'];rows.append([f"{c['length']} / {'causal' if c['causal'] else 'full'}",f"{m['reference']['median_seconds']*1000:.3f}",f"{m['tiled']['median_seconds']*1000:.3f}",f"{m['sdpa_auto']['median_seconds']*1000:.3f}",f"{m['reference']['peak_allocated_bytes']/2**20:.2f}",f"{m['tiled']['peak_allocated_bytes']/2**20:.2f}"])
    table(out/'attention_table.tex',['Length / mask','Dense ms','Tiled ms','SDPA ms','Dense MiB','Tiled MiB'],rows,'lrrrrr')
    rows=[]
    for c in spec['cases']:
        m=c['modes'];rows.append([tex(c['case_id']),f"{m['greedy']['median_seconds']:.3f}",f"{m['speculative_2']['speedup_over_greedy']:.3f}",f"{m['speculative_2']['draft_acceptance_rate']*100:.1f}",f"{m['speculative_4']['speedup_over_greedy']:.3f}",f"{m['speculative_4']['draft_acceptance_rate']*100:.1f}"])
    table(out/'speculative_table.tex',['Case','Greedy (s)','K2 speed','K2 accept \\%','K4 speed','K4 accept \\%'],rows,'lrrrrr')
    history=read('results/finetuning_qlora_v1/training_history.json')
    train=[x for x in history if 'loss'in x];valid=[x for x in history if 'eval_loss'in x]
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.5));axes[0].plot([x['step']for x in train],[x['loss']for x in train],color='#1b6e66',marker='.',label='Training completion loss');axes[0].set(xlabel='Optimizer step',ylabel='Loss',title='Training objective');axes[1].plot([x['epoch']for x in valid],[x['eval_loss']for x in valid],color='#1b6e66',marker='o');axes[1].set(xlabel='Epoch',ylabel='Validation loss',title='Checkpoint selection');axes[1].axvline(1,color='#777',ls='--',lw=1);fig.tight_layout();fig.savefig(out/'training_curve.png');plt.close(fig)
    labels=['Accuracy','Macro F1','Evidence F1','Output valid'];keys=[lambda g:g['classification']['accuracy'],lambda g:g['classification']['macro_f1'],lambda g:g['evidence']['micro']['f1'],lambda g:g['output_valid_rate']]
    x=np.arange(4);fig,ax=plt.subplots(figsize=(6.6,2.5))
    for i,(name,data,color)in enumerate([('Original',before,'#a9b5af'),('Adapted',after,'#1b6e66')]):ax.bar(x+(i-.5)*.34,[fn(data['groups']['overall'])for fn in keys],.34,label=name,color=color)
    ax.set_xticks(x,labels);ax.set(ylim=(0,1.15),ylabel='Validation score');ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(out/'validation_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.6));causal=[c for c in attention['cases']if c['causal']]
    for method,label,color in [('reference','Dense','#8f9c96'),('tiled','Tiled','#1b6e66'),('sdpa_auto','SDPA auto','#303c38')]:
        axes[0].plot([c['length']for c in causal],[c['timing']['methods'][method]['median_seconds']*1000 for c in causal],label=label,color=color,marker='o')
        axes[1].plot([c['length']for c in causal],[c['timing']['methods'][method]['peak_allocated_bytes']/2**20 for c in causal],color=color,marker='o')
    axes[0].set(xlabel='Sequence length (causal)',ylabel='Median milliseconds');axes[0].legend(frameon=False);axes[1].set(xlabel='Sequence length (causal)',ylabel='Peak allocated MiB');fig.tight_layout();fig.savefig(out/'attention_tradeoff.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,2.8));x=np.arange(len(spec['cases']))
    for i,(mode,label,color)in enumerate([('greedy','Greedy','#aab5b0'),('speculative_2','K=2','#1b6e66'),('speculative_4','K=4','#344b43')]):ax.bar(x+(i-1)*.24,[c['modes'][mode]['median_seconds']for c in spec['cases']],.24,color=color,label=label)
    ax.set_xticks(x,[c['case_id']for c in spec['cases']],rotation=20);ax.set_ylabel('Median seconds');ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(out/'speculative_latency.png');plt.close(fig)
    from matplotlib.patches import FancyBboxPatch
    fig,ax=plt.subplots(figsize=(10,3.6));ax.set(xlim=(0,10),ylim=(0,3.6));ax.axis('off')
    boxes=[(.1,2.25,1.7,.9,'Patient record\n+ one criterion'),(2.25,2.25,1.8,.9,'Frozen prompt\n+ tokenization'),(4.5,2.25,2,.9,'Qwen3 decoder\n36 layers + LoRA'),(7,2.25,2.8,.9,'Strict JSON validation\n+ exact source evidence')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor='#f0f4f1',edgecolor='#5b7f72'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for x1,x2 in [(1.85,2.2),(4.1,4.45),(6.55,6.95)]:ax.annotate('',xy=(x2,2.7),xytext=(x1,2.7),arrowprops={'arrowstyle':'->','color':'#1b6e66'})
    ax.text(.2,1.6,'Inside the decoder',fontweight='bold',fontsize=10)
    ax.text(.2,.85,'RoPE positions\n+ causal mask',fontsize=9)
    ax.text(2.6,.85,'Grouped-query attention\n32 query / 8 KV heads',fontsize=9)
    ax.text(5.4,.85,'KV state reused\nor full-prefix reference',fontsize=9)
    ax.text(8,.85,'Low-rank update\nrank 16, scale 32',fontsize=9)
    for x1,x2 in [(1.6,2.5),(4.5,5.3),(7.2,7.9)]:ax.annotate('',xy=(x2,1.03),xytext=(x1,1.03),arrowprops={'arrowstyle':'->','color':'#78998b'})
    fig.tight_layout();fig.savefig(out/'architecture.png');plt.close(fig)
    rows=predictions('results/finetuning_qlora_v1/validation/predictions.jsonl');chosen=[]
    categories=[('Rare negative inclusion',lambda r:r['gold']['label']=='not included' and r['prediction']['label']!='not included'),
        ('Rare exclusion failure',lambda r:r['gold']['label']=='excluded' and r['prediction']['label']!='excluded'),
        ('Missing information mapped to a decision',lambda r:r['gold']['label']=='not enough information' and r['prediction']['label'] in ('included','not excluded')),
        ('Evidence localization failure',lambda r:set(r['gold']['sentence_ids'])!=set(r['prediction']['sentence_ids']))]
    for title,predicate in categories:
        limit=3 if title.startswith('Rare negative')else 2
        matches=[r for r in rows if predicate(r)and r['annotation_id']not in {x[1]['annotation_id']for x in chosen}]
        chosen.extend((title,r)for r in matches[:limit])
    chosen=chosen[:8]
    assert len(chosen)==8
    paragraphs=[]
    for index,(kind,row)in enumerate(chosen,1):
        paragraphs.append(r'\paragraph{'+f'{index}. '+tex(kind)+f' (annotation {row["annotation_id"]})'+'}'+'\n'+
            'Criterion ('+tex(row['criterion_type'])+'): '+tex(row['criterion_text'])+' '+
            'Expert category: '+tex(row['gold']['label'])+'; model category: '+tex(row['prediction']['label'])+'. '+
            'Gold sentence IDs '+tex(row['gold']['sentence_ids'])+'; predicted IDs '+tex(row['prediction']['sentence_ids'])+'. '+
            ('This is a category disagreement; a cited sentence alone does not establish correct interpretation.' if row['gold']['label']!=row['prediction']['label']else 'The category agrees, but the retrieved supporting set differs from the annotation.')+'\n')
    (out/'errors.tex').write_text('\n'.join(paragraphs),encoding='utf-8')
    (ROOT/'reports/validation_error_cases.json').write_text(json.dumps([{'error_group':kind,**row}for kind,row in chosen],indent=2,ensure_ascii=False),encoding='utf-8')
    final=args.final_output/'metrics.json'
    if final.exists():
        metrics=json.loads(final.read_text());assert metrics['split']=='held_out_test' and metrics['complete']
        rows=[]
        for mode in ('before','after'):
            for name,g in metrics['groups'][mode].items():rows.append([name,mode,g['records'],f"{g['classification']['accuracy']:.4f}",f"{g['classification']['macro_f1']:.4f}",f"{g['evidence']['micro']['f1']:.4f}"])
        table(out/'test_table.tex',['Group','Model','N','Accuracy','Macro F1','Evidence F1'],rows,'llrrrr')
        class_rows=[];evidence_rows=[]
        for group in ('overall','inclusion','exclusion'):
            old=metrics['groups']['before'][group];new=metrics['groups']['after'][group]
            for label,values in new['classification']['per_class'].items():
                if label=='__invalid__':continue
                baseline=old['classification']['per_class'][label]
                triplet=lambda value:'/'.join(f'{value[key]:.3f}'for key in ('precision','recall','f1-score'))
                class_rows.append([tex(group+' / '+label),int(values['support']),triplet(baseline),triplet(values)])
            for mode,g in (('before',old),('after',new)):
                e=g['evidence'];evidence_rows.append([group,mode,*[f'{e[kind][key]:.4f}'for kind in ('micro','macro')for key in ('precision','recall','f1')],f"{e['nonempty_gold_macro']['f1']:.4f}"])
        table(out/'test_class_table.tex',['Group / category','N','Before P/R/F1','After P/R/F1'],class_rows,'lrrr')
        table(out/'test_evidence_table.tex',['Group','Model','Micro P','Micro R','Micro F1','Macro P','Macro R','Macro F1','Nonempty F1'],evidence_rows,'llrrrrrrr')
        (out/'test_status.tex').write_text(r'''The final held-out comparison is complete. Table~\ref{tab:test} reports all 206 annotations from the reserved eight patients. Original and adapted phases use the same frozen base, quantization, prompt, greedy decoding and automatic SDPA; the adapter is disabled only for the original phase. The returned predictions were independently checked against the original source inputs and expert targets, re-parsed with the frozen strict parser, and all classification/evidence metrics were recomputed. No subsequent tuning is performed.
\begin{table*}[t]\centering\small\caption{Final held-out test results, separate from development validation.}\label{tab:test}\input{generated/test_table.tex}\end{table*}
Accuracy improves from 0.4078 (84/206) to 0.7427 (153/206), a gain of 33.50 percentage points. Overall macro-F1 rises from 0.3492 to 0.4733. Inclusion accuracy increases from 0.7067 to 0.7600, but inclusion macro-F1 falls slightly from 0.5815 to 0.5802. Exclusion accuracy rises from 0.2366 to 0.7328 and macro-F1 from 0.1905 to 0.3508. Both phases produce valid JSON/categories/citations for every test record; validity is not the same as clinical correctness.

Table~\ref{tab:testclasses} preserves every applicable category and zero-support class. The not-excluded category improves from 2/91 to 89/91 correct, explaining much of the aggregate gain. Not-included recall falls from 3/7 to 1/7, while not-applicable recall falls from 11/15 to 3/15. Included recall is unchanged at 20/27. These regressions prevent describing adaptation as uniformly better. There are no gold excluded examples in this test group, so excluded sensitivity cannot be estimated; its zero score remains in the fixed applicable macro-F1 rather than silently removing the class.
\begin{table*}[t]\centering\footnotesize\caption{Final test category metrics overall and by criterion type. P/R/F1 denotes precision/recall/F1; zero-support classes remain.}\label{tab:testclasses}\input{generated/test_class_table.tex}\end{table*}
Evidence micro-F1 improves from 0.4571 to 0.5714. Adapted micro precision is 0.7925 and recall 0.4468 (42 true positives, 11 false positives and 52 false negatives). Table~\ref{tab:testevidence} includes micro/macro precision, recall and F1, with nonempty-gold-only F1. Of 206 records, 145 have empty gold evidence; consequently macro-F1 of 0.8343 substantially exceeds the nonempty-only value of 0.5553. Exclusion evidence recall is only 0.1250, so improved label accuracy must not obscure weak support retrieval.
\begin{table*}[t]\centering\scriptsize\caption{Final held-out evidence metrics with the explicit empty-set policy. Nonempty F1 is example-macro F1 restricted to nonempty gold.}\label{tab:testevidence}\input{generated/test_evidence_table.tex}\end{table*}
\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{../results/final_test/confusion_after_overall.png}\caption{Selected model's final held-out confusion matrix. Sparse/absent categories and remaining missing-information errors constrain interpretation.}\label{fig:testconfusion}\end{figure}
Figure~\ref{fig:testconfusion} visualizes these remaining errors. The eight examples in the later error-analysis section are drawn from these completed held-out predictions, after every model and prompt choice was fixed; they are analyzed for reporting, not fed back into training.
''',encoding='utf-8')
        test_rows=predictions(args.final_output/'predictions_after.jsonl')
        test_categories=[('Negative inclusion failure',lambda r:r['gold']['label']=='not included'and r['prediction']['label']!='not included',3),
            ('Missing information resolved without support',lambda r:r['gold']['label']=='not enough information'and r['prediction']['label']in('included','not excluded'),2),
            ('Not-applicable criterion misinterpreted',lambda r:r['gold']['label']=='not applicable'and r['prediction']['label']!='not applicable',2),
            ('Incomplete evidence despite matching category',lambda r:r['gold']['label']==r['prediction']['label']and set(r['gold']['sentence_ids'])!=set(r['prediction']['sentence_ids']),1)]
        selected=[]
        for title,predicate,limit in test_categories:
            eligible=[r for r in test_rows if predicate(r)and r['annotation_id']not in {x[1]['annotation_id']for x in selected}]
            selected.extend((title,row)for row in eligible[:limit])
        assert len(selected)==8
        paragraphs=[]
        for index,(kind,row)in enumerate(selected,1):
            paragraphs.append(r'\paragraph{'+f'{index}. '+tex(kind)+f' (annotation {row["annotation_id"]})'+'}\n'+
                'Criterion ('+tex(row['criterion_type'])+'): '+tex(row['criterion_text'])+' '+
                'Expert: '+tex(row['gold']['label'])+'; model: '+tex(row['prediction']['label'])+'. '+
                'Gold IDs '+tex(row['gold']['sentence_ids'])+'; predicted IDs '+tex(row['prediction']['sentence_ids'])+'. '+
                ('The category conflicts with the reference; a citation does not by itself justify the interpretation.'if row['gold']['label']!=row['prediction']['label']else'The category agrees but the source evidence set is incomplete or differs from the reference.')+'\n')
        (out/'errors.tex').write_text('\n'.join(paragraphs),encoding='utf-8')
        (ROOT/'reports/final_test_error_cases.json').write_text(json.dumps([{'error_group':kind,**row}for kind,row in selected],indent=2,ensure_ascii=False),encoding='utf-8')
    else:
        (out/'test_status.tex').write_text(r'\textbf{Final test evaluation is pending.} No held-out accuracy, F1 or clinical generalization result is claimed in this draft. All classification tables currently report validation. The paired final-evaluation runner and Colab cell are supplied; their returned artifacts must be incorporated before claiming completion of held-out assessment.',encoding='utf-8')
    screenshot=ROOT/'artifacts/app_browser_final/desktop-fold.jpg'
    if screenshot.exists():shutil.copyfile(screenshot,out/'application.jpg')
    print('Generated experiment tables, plots, architecture, eight development errors and (when present) eight held-out errors and final metrics.')


if __name__=='__main__':main()
