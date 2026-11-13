/* Uses the existing workspace's escaping, API client and review editor. */
const reviewAgent = (() => {
  let timer = null;
  const terminal = new Set(['completed','failed','canceled','interrupted','needs_manual_reading']);
  const statusNames = {queued:'в очереди',running:'проверяется',completed:'черновик готов',failed:'ошибка',canceled:'остановлено',interrupted:'прервано',needs_manual_reading:'нужно прочитать вручную'};
  const eventNames = {queued:'Очередь',started:'Проверка началась',model_turn:'Обращение к модели',request_retry:'Повтор запроса к модели',text_delta:'Сообщение агента',tool_call:'Вызов инструмента',tool_result:'Результат инструмента',tool_error:'Инструмент вернул ошибку',completed:'Черновик сохранён',needs_manual_reading:'Нужно ручное чтение',cancel_requested:'Остановка запрошена',canceled:'Остановлено',failed:'Проверка прервана',interrupted:'Сервер перезапущен'};
  const toolNames = {read_context:'Контекст и критерии',read_page:'Чтение страницы',check_logic:'Таблица истинности',calculate_sets:'Вычисление множеств',submit_draft:'Валидация черновика',request_manual_reading:'Запрос ручного чтения'};
  const resultSummary = (name,r) => {
    if(!r)return '';
    if(r.error)return r.error;
    if(name==='read_context')return `Задач: ${r.problems?.length||0}. Источников: ${Object.keys(r.sources||{}).length}.`;
    if(name==='read_page')return `${({work:'Работа',statement:'Условие',rubric:'Критерии'})[r.source]||r.source}: страница ${r.page} из ${r.page_count}${r.crop?' · увеличенный фрагмент':''}.`;
    if(name==='check_logic')return r.counterexample!==null?`Найден контрпример: ${Object.entries(r.counterexample||{}).map(([k,v])=>k+' = '+(v?'истина':'ложь')).join(', ')}. Утверждение не выполняется.`:`Проверено наборов: ${r.rows?.length||0}. ${r.equivalent!==undefined?'Формулы равносильны.':'Формула является тавтологией.'}`;
    if(name==='calculate_sets'){const format=v=>Array.isArray(v)?'{'+v.map(format).join(', ')+'}':typeof v==='boolean'?(v?'да':'нет'):String(v);return 'Результат вычисления: '+format(r.result)+'.'}
    if(name==='submit_draft'&&r.validated)return `Покрытие задач и основания проверены. Сумма: ${r.total}. Черновик будет сохранён после завершения ответа модели.`;
    if(name==='request_manual_reading')return r.reason||'';
    return '';
  };
  const stopPolling = () => { clearTimeout(timer); timer=null; };

  async function settings() {
    stopPolling();
    const s = await api('/api/model/status');
    const profile = s.profiles.find(p=>p.id===s.active_profile);
    app.innerHTML = `<div class="grid"><article class="card span-6"><p class="eyebrow dark">Модель и подключение</p><h2>Агент проверки</h2><p class="muted agent-intro">Агент читает материалы, выбирает инструменты и готовит черновик. Итог утверждаете вы.</p><form id="model-settings"><label>Режим<select name="provider"><option value="subscription">Подписка ChatGPT</option><option value="api">OpenAI API · отдельная оплата</option>${s.demo_allowed?'<option value="replay">Сценарная проверка · без LLM</option>':''}</select></label><p id="provider-note" class="history"></p><label>Модель<div class="model-picker"><input name="model" value="${esc(s.model)}" placeholder="Выберите доступную модель" list="model-catalog"><button type="button" class="secondary" id="load-models">Получить список</button></div><datalist id="model-catalog"></datalist></label><div class="actions"><button>Сохранить режим</button><button type="button" class="secondary" id="probe-model">Проверить подключение</button></div></form><div id="probe-result" aria-live="polite"></div></article><article class="card span-6"><h2>Подключение ChatGPT</h2><p class="muted agent-intro">Вход через официальный OAuth. Разрешение использовать подписку проверяется отдельно от самого входа.</p><div class="history">${profile?`${esc(profile.email||'Аккаунт ChatGPT')}<br>${profile.connected?(profile.plan_enabled?'Подписка разрешена':'Нужно разрешение использования плана'):'Требуется повторный вход'}`:'Аккаунт ещё не подключён'}</div><div class="actions agent-spaced"><button id="connect-chatgpt">Continue with ChatGPT</button><button id="refresh-account" class="secondary">Обновить статус</button>${profile?'<button id="logout-chatgpt" class="secondary">Выйти</button>':''}</div>${s.profiles.length?`<label class="agent-spaced">Сохранённые подключения<select id="saved-profiles">${s.profiles.map(p=>`<option value="${esc(p.id)}" ${p.id===s.active_profile?'selected':''}>${esc(p.email||p.subject)} · ${esc(p.id.slice(-8))}</option>`).join('')}</select></label><div class="actions agent-spaced"><button id="select-profile" class="secondary">Выбрать</button><button id="reconnect-profile" class="secondary">Повторить вход</button></div>`:''}<p class="muted small agent-spaced">Платный API включается только выбором режима. Для него задайте OPENAI_API_KEY при запуске сервера. Автоматическое переключение оплаты выключено.</p><a class="file-link small" href="https://chatgpt.com/settings/usage" target="_blank" rel="noreferrer">Использование плана в ChatGPT</a></article></div>`;
    const form = $('#model-settings');
    form.provider.value=s.provider;
    const note=()=>{$('#provider-note').textContent={subscription:'Запросы используют разрешённую квоту выбранного ChatGPT-аккаунта.',api:'Запросы оплачиваются отдельно по API-ключу сервера.',replay:'Без LLM и сетевых запросов. Предопределённый сценарий проверяет инструменты и интерфейс; это не демонстрация AI-агента.'}[form.provider.value]};
    note(); form.provider.onchange=note;
    form.onsubmit=async e=>{e.preventDefault();try{await post('/api/model/config',{provider:form.provider.value,model:form.model.value});say('Подключение сохранено.');}catch(error){say(error.message,true)}};
    $('#load-models').onclick=async e=>{e.target.disabled=true;try{await post('/api/model/config',{provider:form.provider.value,model:form.model.value});const models=await api('/api/model/models');$('#model-catalog').innerHTML=models.map(m=>`<option value="${esc(m.id)}">${esc(m.name)}</option>`).join('');if(!form.model.value&&models.length)form.model.value=models[0].id;say(`Доступно моделей: ${models.length}. Выберите и сохраните модель.`)}catch(error){say(error.message,true)}finally{e.target.disabled=false}};
    $('#probe-model').onclick=async e=>{e.target.disabled=true;$('#probe-result').innerHTML='<p class="muted agent-spaced">Проверяю изображение, вызов инструмента и использование его результата…</p>';try{await post('/api/model/config',{provider:form.provider.value,model:form.model.value});await post('/api/model/probe');$('#probe-result').innerHTML='<p class="history agent-spaced">Проверка пройдена: изображение прочитано, инструмент вызван, его результат использован, оба ответа завершены.</p>'}catch(error){$('#probe-result').textContent=error.message}finally{e.target.disabled=false}};
    const connect=async profileId=>{try{const result=await post('/api/model/auth/start',profileId?{profile_id:profileId}:{});window.location.assign(result.authorization_url)}catch(error){say(error.message,true)}};
    $('#connect-chatgpt').onclick=()=>connect();
    $('#refresh-account').onclick=settings;
    $('#reconnect-profile')?.addEventListener('click',()=>connect($('#saved-profiles').value));
    $('#select-profile')?.addEventListener('click',async()=>{try{await post('/api/model/auth/select',{profile_id:$('#saved-profiles').value});await settings()}catch(error){say(error.message,true)}});
    $('#logout-chatgpt')?.addEventListener('click',async()=>{try{const result=await post('/api/model/auth/logout');await settings();say(result.remote_revocation_confirmed?'Подключение отключено.':'Локальный выход выполнен. Удалённый отзыв не подтверждён; отключить приложение можно в настройках ChatGPT.')}catch(error){say(error.message,true)}});
  }

  async function mount(homeworkId, records) {
    stopPolling();
    app.querySelectorAll('[data-review]').forEach(row=>{
      const r=records.find(item=>item.id===row.dataset.review);
      const cell=document.createElement('td');
      cell.innerHTML=`<input type="checkbox" class="review-select" aria-label="Выбрать ${esc(r.full_name)}" value="${r.id}" ${!r.submission_id||r.status==='approved'?'disabled':''}>`;
      cell.onclick=e=>e.stopPropagation();row.prepend(cell);
    });
    const table=app.querySelector('[data-review]')?.closest('table');
    if(table){const th=document.createElement('th');th.innerHTML='<input type="checkbox" id="select-all-reviews" aria-label="Выбрать все доступные работы">';table.querySelector('thead tr').prepend(th);$('#select-all-reviews').onchange=e=>app.querySelectorAll('.review-select:not(:disabled)').forEach(c=>c.checked=e.target.checked)}
    const section=document.createElement('article');section.className='card span-12 agent-panel';section.id='agent-panel';
    section.innerHTML=`<div class="row"><div><p class="eyebrow dark">Автоматическая проверка</p><h2>Действия агента</h2><p class="muted small" id="agent-mode">Загружаю подключение…</p></div><div class="actions"><button id="check-selected">Проверить выбранные</button><button class="secondary" id="agent-settings">Подключение</button></div></div><div id="agent-runs" class="agent-runs" aria-live="polite"></div><details id="agent-run-history" class="run-history" hidden><summary>История запусков</summary><div id="agent-past-runs" class="agent-runs"></div></details><div id="agent-timeline"></div>`;
    $('#review-content').prepend(section);
    let selectedRun=null, cursor=0, knownCompleted=new Set(), first=true;
    let currentRuns=[], latestRunIds=new Set();
    const selectRun=id=>{selectedRun=id;cursor=0;$('#agent-timeline').innerHTML='';};
    $('#agent-run-history').ontoggle=()=>{
      if($('#agent-run-history').open||!selectedRun||latestRunIds.has(selectedRun))return;
      const selected=currentRuns.find(r=>r.id===selectedRun);
      const latest=currentRuns.find(r=>r.review_id===selected?.review_id&&latestRunIds.has(r.id));
      selectRun(latest?.id||currentRuns[0]?.id||null);
      poll();
    };
    const mode=await api('/api/model/status');
    $('#agent-mode').textContent=mode.provider==='replay'?'СЦЕНАРНЫЙ РЕЖИМ · без LLM':`${mode.provider==='subscription'?'Подписка ChatGPT':'Платный OpenAI API'} · ${mode.model||'модель не выбрана'}`;
    if(mode.provider==='replay')section.classList.add('scripted-panel');
    $('#agent-settings').onclick=()=>render('agent');
    $('#check-selected').onclick=async e=>{const ids=[...app.querySelectorAll('.review-select:checked')].map(c=>c.value);if(!ids.length)return say('Отметьте работы флажками в таблице.',true);e.target.disabled=true;try{const runs=await post('/api/agent/runs',{review_ids:ids});selectedRun=runs[0].id;cursor=0;$('#agent-timeline').innerHTML='';say(`В очередь добавлено работ: ${runs.length}.`);await poll()}catch(error){say(error.message,true)}finally{e.target.disabled=false}};
    async function poll(){
      if(state.view!=='reviews'||state.selectedHomework!==homeworkId||!$('#agent-runs'))return;
      clearTimeout(timer);
      try{
        const runs=await api('/api/agent/runs?homework_id='+encodeURIComponent(homeworkId));
        currentRuns=runs;
        const seenReviews=new Set(), latest=[], previous=[];
        for(const run of runs){
          if(seenReviews.has(run.review_id))previous.push(run);
          else{seenReviews.add(run.review_id);latest.push(run)}
        }
        latestRunIds=new Set(latest.map(r=>r.id));
        if(!selectedRun&&latest.length)selectedRun=latest[0].id;
        if(selectedRun&&!latestRunIds.has(selectedRun)&&!$('#agent-run-history').open){
          const previousSelection=runs.find(r=>r.id===selectedRun);
          selectRun(latest.find(r=>r.review_id===previousSelection?.review_id)?.id||latest[0]?.id||null);
        }
        const renderRun=r=>{const record=records.find(x=>x.id===r.review_id);return `<div class="run-item ${r.id===selectedRun?'active':''}"><button class="run-open" data-run="${r.id}"><strong>${esc(record?.full_name||'Работа')}</strong><span class="badge ${esc(r.status)}">${esc(statusNames[r.status]||r.status)}</span><span class="small muted">${r.provider==='replay'?'Без LLM · сценарий':esc(r.model)}</span></button>${!terminal.has(r.status)?`<button class="secondary" data-cancel="${r.id}" ${r.cancel_requested?'disabled':''}>${r.cancel_requested?'Останавливается…':'Остановить'}</button>`:''}${terminal.has(r.status)?`<button class="secondary" data-result="${r.review_id}">Открыть работу</button>`:''}</div>`};
        $('#agent-runs').innerHTML=latest.length?latest.map(renderRun).join(''):'<p class="muted agent-spaced">Выберите работы и запустите проверку. Здесь появятся вызовы инструментов и их результаты.</p>';
        const history=$('#agent-run-history');
        history.hidden=!previous.length;
        $('summary',history).textContent=`История запусков (${previous.length})`;
        $('#agent-past-runs').innerHTML=previous.map(renderRun).join('');
        if(first){runs.filter(r=>r.status==='completed').forEach(r=>knownCompleted.add(r.id));first=false}
        const newly=runs.filter(r=>r.status==='completed'&&!knownCompleted.has(r.id));
        if(newly.length){const fresh=await api('/api/homeworks/'+homeworkId+'/records');for(const item of fresh){const index=records.findIndex(x=>x.id===item.id);if(index>=0)records[index]=item;const row=app.querySelector(`[data-review="${item.id}"]`);if(row){const cells=row.querySelectorAll('td');cells[3].innerHTML=badge(item.status)+'<br>'+reviewTotals(item)}}newly.forEach(r=>knownCompleted.add(r.id))}
        app.querySelectorAll('[data-run]').forEach(b=>b.onclick=()=>{selectRun(b.dataset.run);poll()});
        app.querySelectorAll('[data-cancel]').forEach(b=>b.onclick=async()=>{try{await post('/api/agent/runs/'+b.dataset.cancel+'/cancel');await poll()}catch(error){say(error.message,true)}});
        app.querySelectorAll('[data-result]').forEach(b=>b.onclick=()=>openReview(b.dataset.result));
        if(selectedRun){const run=await api('/api/agent/runs/'+selectedRun+'?after='+cursor);if(!cursor)$('#agent-timeline').innerHTML='<div class="section-title">Журнал действий</div><div class="timeline"></div>';for(const event of run.events){const data=event.data;if(event.kind==='text_delta'&&!data.text)continue;const text=data.text||data.message;const title=(data.tool?toolNames[data.tool]||data.tool:eventNames[event.kind]||event.kind);const detail=data.arguments||data.result;const summary=event.kind==='tool_result'||event.kind==='tool_error'?resultSummary(data.tool,data.result):'';const prior=$('#agent-timeline .timeline').lastElementChild;if(event.kind==='text_delta'&&prior?.classList.contains('text_delta')){prior.querySelector('p').textContent+=data.text;continue}$('#agent-timeline .timeline').insertAdjacentHTML('beforeend',`<div class="timeline-event ${esc(event.kind)}"><div class="timeline-title"><strong>${esc(title)}</strong><time>${esc(new Date(event.created_at).toLocaleTimeString('ru-RU'))}</time></div>${text?`<p>${esc(text)}</p>`:''}${data.step?`<span class="muted small">Шаг ${data.step}</span>`:''}${summary?`<p>${esc(summary)}</p>`:''}${detail?`<details><summary>${event.kind==='tool_call'?'Параметры вызова':'Данные результата'}</summary><pre>${esc(JSON.stringify(detail,null,2))}</pre></details>`:''}${data.has_image?'<span class="badge">Страница передана модели изображением</span>':''}</div>`)}cursor=run.cursor;if(run.error?.message&&!$('#run-error'))$('#agent-timeline').insertAdjacentHTML('afterbegin',`<p id="run-error" class="history warning">${esc(run.error.message)}</p>`)}
      }catch(error){const node=$('#agent-mode');if(node)node.textContent=error.message}
      if(state.view==='reviews'&&state.selectedHomework===homeworkId)timer=setTimeout(poll,1200);
    }
    await poll();
  }

  async function detail(review, homework){
    modal.classList.add('review-dialog');
    const body=$('.modal-body',modal), editor=document.createElement('div');editor.className='review-editor';
    while(body.firstChild)editor.append(body.firstChild);
    body.classList.add('review-split');body.append(editor);
    const preview=document.createElement('aside');preview.className='work-preview';
    preview.innerHTML=review.submission_id?`<div class="row"><strong>Оригинал работы</strong><div class="actions"><button type="button" class="secondary" id="page-back" aria-label="Предыдущая страница">←</button><span id="page-number">1</span><button type="button" class="secondary" id="page-next" aria-label="Следующая страница">→</button></div></div><img id="work-page" alt="Страница работы студента"><p class="muted small">Нажмите страницу в основаниях, чтобы перейти к соответствующему фрагменту.</p>`:'<p class="muted">Работа ещё не импортирована.</p>';
    body.prepend(preview);
    const form=$('#review-form',modal);form.dataset.revision=review.revision;
    $('.review-top',editor).after(form);
    const lateness=document.createElement('details');lateness.innerHTML='<summary>Коэффициент просрочки</summary>';
    lateness.append(form.late_multiplier.closest('label'),form.late_reason.closest('label'));
    form.querySelector('.actions').before(lateness);
    const versions=document.createElement('details');versions.className='review-versions';versions.innerHTML='<summary>История версий черновика</summary><div class="versions-content"></div>';editor.append(versions);
    versions.ontoggle=async()=>{if(!versions.open)return;try{const rows=await api('/api/records/'+review.id+'/versions');$('.versions-content',versions).innerHTML=rows.map(v=>{const p=v.payload;return `<details><summary>Версия ${v.revision} · ${esc(v.action)} · ${esc(new Date(v.created_at).toLocaleString('ru-RU'))}</summary><pre>${esc(JSON.stringify({scores:JSON.parse((p.status==='approved'?p.approved_scores_json:p.suggested_scores_json)||'[]'),total:p.status==='approved'?p.approved_total:p.suggested_total,student_comment:p.student_comment,service_comment:p.service_comment},null,2))}</pre></details>`}).join('')||'<p class="muted">Сохранённых версий пока нет.</p>'}catch(error){$('.versions-content',versions).textContent=error.message}};
    if(review.status!=='approved'&&review.submission_id){const button=document.createElement('button');button.type='button';button.textContent='Проверить агентом';button.className='secondary';$('.review-top .actions',editor).prepend(button);button.onclick=async()=>{try{await post('/api/agent/runs',{review_ids:[review.id]});modal.close();await loadReviews(homework.id)}catch(error){say(error.message,true)}}}
    if(!review.submission_id)return;
    let page=1,count=1;
    const show=p=>{page=Math.max(1,Math.min(count,p));$('#page-number',preview).textContent=`${page} / ${count}`;$('#work-page',preview).src='/api/records/'+review.id+'/page?page='+page;$('#page-back',preview).disabled=page===1;$('#page-next',preview).disabled=page===count};
    try{count=(await api('/api/records/'+review.id+'/pages')).pages;show(1)}catch(error){$('#work-page',preview).alt=error.message}
    $('#page-back',preview).onclick=()=>show(page-1);$('#page-next',preview).onclick=()=>show(page+1);
    const jumps=document.createElement('div');jumps.className='actions evidence-jumps';
    jumps.innerHTML=(review.evidence||[]).filter(e=>e.source==='work').map(e=>`<button type="button" class="secondary" data-page="${Number(e.page)||1}" title="${esc(e.note)}">Стр. ${esc(e.page)}</button>`).join('');preview.append(jumps);jumps.querySelectorAll('[data-page]').forEach(b=>b.onclick=()=>show(Number(b.dataset.page)));
  }
  return {settings,mount,detail,stopPolling};
})();
window.reviewAgent = reviewAgent;
