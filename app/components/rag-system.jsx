import { useState, useRef } from "react";

// ─── RAG UTILITIES ─────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  'the','a','an','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','will','would','could','should','may','might','shall','can',
  'to','of','in','for','on','with','at','by','from','as','into','through',
  'during','before','after','above','below','between','out','off','over','under',
  'then','once','here','there','when','where','why','how','all','both','each',
  'few','more','other','some','such','no','nor','not','so','yet','but','and',
  'or','because','if','while','although','than','that','this','these','those',
  'i','you','he','she','it','we','they','what','which','who','whom',
  'his','her','its','my','your','their','our','also','just','very','about'
]);

function tokenize(text) {
  return text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS.has(t));
}

function chunkText(text, chunkSize, overlapRatio = 0.15) {
  const words = text.split(/\s+/).filter(Boolean);
  const overlap = Math.max(1, Math.floor(chunkSize * overlapRatio));
  const chunks = [];
  let i = 0;
  while (i < words.length) {
    const end = Math.min(i + chunkSize, words.length);
    chunks.push(words.slice(i, end).join(' '));
    if (end >= words.length) break;
    i += chunkSize - overlap;
  }
  return chunks;
}

function buildChunkIndex(documents, chunkSize) {
  const chunks = [];
  for (const doc of documents) {
    const docChunks = chunkText(doc.content, chunkSize);
    for (let i = 0; i < docChunks.length; i++) {
      chunks.push({ id: `${doc.id}-${i}`, docId: doc.id, docTitle: doc.title, text: docChunks[i], chunkIndex: i });
    }
  }
  return chunks;
}

function computeTFIDF(query, chunks) {
  const queryTokens = tokenize(query);
  if (!queryTokens.length) return chunks.map((c, i) => ({ ...c, score: 0, rank: i }));
  const chunkTokens = chunks.map(c => tokenize(c.text));
  const N = chunks.length;
  const df = {};
  for (const tokens of chunkTokens)
    for (const t of new Set(tokens)) df[t] = (df[t] || 0) + 1;

  const scored = chunkTokens.map((tokens, i) => {
    const tf = {};
    for (const t of tokens) tf[t] = (tf[t] || 0) + 1;
    const len = tokens.length || 1;
    let score = 0;
    for (const qt of queryTokens) {
      const termTF = (tf[qt] || 0) / len;
      const termIDF = Math.log((N + 1) / ((df[qt] || 0) + 1)) + 1;
      score += termTF * termIDF;
    }
    return { ...chunks[i], score };
  });

  return scored.sort((a, b) => b.score - a.score);
}

// ─── SAMPLE DATA ───────────────────────────────────────────────────────────────

const SAMPLE_DOCS = [
  {
    id: 'doc1', title: 'Neural Networks & Deep Learning',
    content: `A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input so the network generates the best possible result without needing to redesign the output criteria.

Neural networks are composed of layers of nodes, containing an input layer, one or more hidden layers, and an output layer. Each node, or artificial neuron, connects to another and has an associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer.

Training neural networks requires large datasets and significant computational power. The backpropagation algorithm adjusts the weights by calculating the gradient of the loss function with respect to each weight. This process is repeated until the network achieves acceptable performance.

Deep learning refers to neural networks with multiple hidden layers. These architectures have proven remarkably effective at image recognition, natural language processing, and game playing. Convolutional neural networks (CNNs) excel at image tasks, while transformers dominate natural language tasks. The attention mechanism in transformers allows the model to focus on different parts of the input when generating each part of the output.`
  },
  {
    id: 'doc2', title: 'History of the Internet',
    content: `The Internet originated in the United States government-sponsored computer network called ARPANET in the 1960s. ARPANET was the world's first operational packet-switching network. The network was funded by the Advanced Research Projects Agency (ARPA) of the U.S. Department of Defense to enable researchers to share resources.

In the early 1970s, the development of TCP/IP protocols enabled different networks to communicate with each other, laying the groundwork for the modern Internet. Vinton Cerf and Bob Kahn are credited with designing the fundamental Internet protocol suite. The Domain Name System (DNS) was introduced in 1983 to replace numerical IP addresses with human-readable names.

Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN. The web is a system of Internet servers that support specially formatted documents interconnected via hyperlinks. Mosaic, the first popular graphical web browser, was released in 1993 and sparked a massive expansion of Internet usage among the general public.

The commercialization of the Internet began in the mid-1990s. Companies like Amazon, eBay, and Yahoo were founded during this period. The dot-com bubble of the late 1990s saw enormous investment followed by a crash in 2000–2001. Despite this, the Internet continued to grow and transform global society fundamentally.`
  },
  {
    id: 'doc3', title: 'Climate Change & Renewable Energy',
    content: `Climate change refers to long-term shifts in temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil, and gas. Burning fossil fuels generates greenhouse gas emissions that trap the sun's heat and raise global temperatures.

Carbon dioxide and methane are the primary greenhouse gases. Carbon dioxide is released through burning fossil fuels and deforestation. Methane is emitted from livestock, landfills, and oil and gas operations. The concentration of carbon dioxide in the atmosphere has increased from about 280 parts per million before industrialization to over 420 parts per million today.

The consequences of climate change include rising sea levels, more intense weather events, loss of biodiversity, and threats to food security. The IPCC has warned that limiting global warming to 1.5 degrees Celsius requires rapid and far-reaching transitions in energy, land, transport, and industrial systems.

Renewable energy sources such as solar and wind power are increasingly cost-competitive with fossil fuels. Energy efficiency improvements, electrification of transportation, and carbon capture technologies are also critical tools. International agreements like the Paris Agreement aim to coordinate global action on climate mitigation.`
  }
];

const SAMPLE_QUESTIONS = [
  'How does backpropagation work in neural networks?',
  'Who invented the World Wide Web and when?',
  'What are the main causes of climate change?',
  'What is the difference between CNN and transformer architectures?',
  'How did ARPANET lead to the modern internet?'
];

const DOC_COLORS = ['#f59e0b', '#34d399', '#60a5fa'];

// ─── COMPONENT ─────────────────────────────────────────────────────────────────

export default function RAGLesson() {
  const [docs, setDocs] = useState(SAMPLE_DOCS);
  const [query, setQuery] = useState('');
  const [stage, setStage] = useState('idle');
  const [allChunks, setAllChunks] = useState([]);
  const [scored, setScored] = useState([]);
  const [topK, setTopK] = useState(3);
  const [chunkSize, setChunkSize] = useState(100);
  const [answer, setAnswer] = useState('');
  const [ragPrompt, setRagPrompt] = useState('');
  const [activeTab, setActiveTab] = useState('explainer');
  const [showAddDoc, setShowAddDoc] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newContent, setNewContent] = useState('');
  const [error, setError] = useState('');
  const [currentQuery, setCurrentQuery] = useState('');

  const docColorMap = Object.fromEntries(docs.map((d, i) => [d.id, DOC_COLORS[i % DOC_COLORS.length]]));

  const runRAG = async (q = query) => {
    if (!q.trim() || (stage !== 'idle' && stage !== 'done')) return;
    setError(''); setAnswer(''); setRagPrompt(''); setCurrentQuery(q);

    // 1 — CHUNK
    setStage('chunking'); setActiveTab('chunks');
    await sleep(350);
    const chunks = buildChunkIndex(docs, chunkSize);
    setAllChunks(chunks);

    // 2 — RETRIEVE
    setStage('retrieving'); setActiveTab('retrieval');
    await sleep(350);
    const ranked = computeTFIDF(q, chunks);
    setScored(ranked);

    // 3 — BUILD PROMPT
    setStage('generating'); setActiveTab('answer');
    await sleep(150);
    const topChunks = ranked.slice(0, topK);
    const context = topChunks.map((c, i) =>
      `[Source ${i + 1} — ${c.docTitle}, chunk ${c.chunkIndex + 1}]\n${c.text}`
    ).join('\n\n---\n\n');

    const prompt = `You are a helpful assistant. Answer the question using ONLY the provided context. If the context is insufficient, say so clearly. Cite your sources by number.

CONTEXT:
${context}

---

QUESTION: ${q}`;
    setRagPrompt(prompt);

    // 4 — GENERATE
    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 1000,
          messages: [{ role: 'user', content: prompt }]
        })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error.message);
      setAnswer(data.content?.map(b => b.text || '').join('') || '—');
      setStage('done');
    } catch (e) {
      setError(e.message);
      setStage('done');
    }
  };

  const addDoc = () => {
    if (!newTitle.trim() || !newContent.trim()) return;
    setDocs(prev => [...prev, { id: 'doc' + Date.now(), title: newTitle, content: newContent }]);
    setNewTitle(''); setNewContent(''); setShowAddDoc(false);
  };

  const maxScore = scored.length > 0 ? Math.max(scored[0].score, 0.0001) : 1;
  const isRunning = stage !== 'idle' && stage !== 'done';

  const STAGES = [
    { key: 'chunking', label: 'Chunking' },
    { key: 'retrieving', label: 'Retrieval' },
    { key: 'generating', label: 'Generation' },
    { key: 'done', label: 'Done' },
  ];
  const stageIdx = STAGES.findIndex(s => s.key === stage);

  return (
    <div style={s.root}>
      {/* ── TOP HEADER ── */}
      <header style={s.header}>
        <div>
          <div style={s.logo}>RAG_SYSTEM<span style={{ opacity: 0.35 }}>.jsx</span></div>
          <div style={s.subtitle}>Retrieval-Augmented Generation · Interactive Lesson</div>
        </div>

        {/* pipeline pill */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {STAGES.map((st, i) => {
            const active = stageIdx === i && stage !== 'done' && stage !== 'idle';
            const done = stageIdx > i || stage === 'done';
            return (
              <div key={st.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{
                  ...s.pill,
                  borderColor: done || active ? '#f59e0b55' : '#ffffff11',
                  background: done || active ? '#f59e0b11' : 'transparent',
                  color: active ? '#f59e0b' : done ? '#f59e0b88' : '#ffffff33',
                }}>
                  {active && <span style={s.dot} />}
                  {st.label}
                </div>
                {i < STAGES.length - 1 && <span style={{ color: '#ffffff22', fontSize: 10 }}>›</span>}
              </div>
            );
          })}
        </div>
      </header>

      <div style={s.layout}>
        {/* ── LEFT SIDEBAR ── */}
        <aside style={s.sidebar}>
          <div style={s.sideSection}>
            <div style={s.sideLabel}>CORPUS <span style={{ opacity: 0.45 }}>{docs.length} docs</span></div>
            <button style={s.addBtn} onClick={() => setShowAddDoc(v => !v)}>
              {showAddDoc ? '✕ cancel' : '+ add doc'}
            </button>
          </div>

          {showAddDoc && (
            <div style={s.addForm}>
              <input value={newTitle} onChange={e => setNewTitle(e.target.value)}
                placeholder="Title" style={s.input} />
              <textarea value={newContent} onChange={e => setNewContent(e.target.value)}
                placeholder="Paste content..." rows={6} style={{ ...s.input, resize: 'vertical' }} />
              <button onClick={addDoc} style={s.ingestBtn}>INGEST DOCUMENT</button>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
            {docs.map((doc, i) => {
              const color = DOC_COLORS[i % DOC_COLORS.length];
              const wc = doc.content.split(/\s+/).length;
              const nc = buildChunkIndex([doc], chunkSize).length;
              return (
                <div key={doc.id} style={{ ...s.docCard, borderLeftColor: color, borderColor: color + '22', background: color + '08' }}>
                  <div style={{ fontSize: 12, fontWeight: 600, lineHeight: 1.4, marginBottom: 4 }}>{doc.title}</div>
                  <div style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.45 }}>
                    {wc} words · {nc} chunks @ size={chunkSize}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Config */}
          <div style={{ marginTop: 24, paddingTop: 16, borderTop: '1px solid #1e1e28' }}>
            <div style={s.sideLabel}>CONFIG</div>
            <label style={s.configLabel}>Chunk size: <strong style={{ color: '#f59e0b' }}>{chunkSize} words</strong></label>
            <input type="range" min={50} max={250} step={25} value={chunkSize}
              onChange={e => setChunkSize(+e.target.value)} style={s.range} />
            <label style={s.configLabel}>Top-K: <strong style={{ color: '#f59e0b' }}>{topK} chunks</strong></label>
            <input type="range" min={1} max={6} step={1} value={topK}
              onChange={e => setTopK(+e.target.value)} style={s.range} />
          </div>
        </aside>

        {/* ── MAIN ── */}
        <main style={s.main}>
          {/* Query bar */}
          <div style={s.queryBar}>
            <div style={{ flex: 1, position: 'relative' }}>
              <span style={s.queryArrow}>›</span>
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && runRAG()}
                placeholder="Ask a question about your documents…"
                style={s.queryInput}
              />
            </div>
            <button onClick={() => runRAG()} disabled={isRunning} style={{ ...s.runBtn, opacity: isRunning ? 0.45 : 1 }}>
              {stage === 'chunking' ? 'CHUNKING…' : stage === 'retrieving' ? 'RETRIEVING…' : stage === 'generating' ? 'GENERATING…' : 'RUN RAG ↵'}
            </button>
          </div>

          {/* Tabs */}
          <div style={s.tabs}>
            {[
              { key: 'explainer', label: 'How RAG Works' },
              { key: 'chunks', label: `Chunks${allChunks.length ? ' (' + allChunks.length + ')' : ''}` },
              { key: 'retrieval', label: 'Retrieval Scores' },
              { key: 'answer', label: 'Answer' },
              { key: 'prompt', label: 'RAG Prompt' },
            ].map(tab => (
              <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
                ...s.tabBtn,
                color: activeTab === tab.key ? '#f59e0b' : '#ffffff44',
                borderBottom: activeTab === tab.key ? '2px solid #f59e0b' : '2px solid transparent',
              }}>{tab.label}</button>
            ))}
          </div>

          {/* ── TAB: EXPLAINER ── */}
          <div style={{ ...s.tabContent, display: activeTab === 'explainer' ? 'block' : 'none' }}>
            <div style={{ maxWidth: 680 }}>
              <h2 style={s.h2}>What is RAG?</h2>
              <p style={s.body}>
                LLMs are trained on a static snapshot of the world — they can't access your private docs, recent news,
                or proprietary data. <strong>Retrieval-Augmented Generation</strong> solves this by fetching relevant
                information at query-time and injecting it into the prompt as context. The model reasons over
                fresh data without ever being retrained.
              </p>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, margin: '20px 0' }}>
                {[
                  { n: '01', title: 'Ingest', body: `Load your documents. Here we have ${docs.length} docs totalling ~${docs.reduce((s, d) => s + d.content.split(/\s+/).length, 0)} words.` },
                  { n: '02', title: 'Chunk', body: `Split each doc into overlapping windows of ${chunkSize} words (~15% overlap). Overlap preserves context at boundaries.` },
                  { n: '03', title: 'Index', body: 'Build a retrieval index. Production systems use dense vector embeddings (semantic search). This demo uses TF-IDF (keyword overlap).' },
                  { n: '04', title: 'Retrieve', body: `Score all ${buildChunkIndex(docs, chunkSize).length} chunks against the query. Return the top-${topK} by score.` },
                  { n: '05', title: 'Generate', body: 'Inject top-K chunks as context into a prompt. Claude answers strictly from that context — grounded, not hallucinated.' },
                ].map(item => (
                  <div key={item.n} style={s.stepCard}>
                    <span style={s.stepNum}>{item.n}</span>
                    <div>
                      <div style={s.stepTitle}>{item.title}</div>
                      <div style={{ ...s.body, margin: 0 }}>{item.body}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div style={s.callout}>
                <div style={s.calloutLabel}>TF-IDF vs Vector Embeddings</div>
                <p style={{ margin: 0, ...s.body }}>
                  TF-IDF scores chunks by term overlap — fast and interpretable, but misses synonyms and meaning.
                  Production RAG systems use <strong>dense embeddings</strong> (e.g. <code>text-embedding-3-small</code>)
                  which map text to high-dimensional vectors where semantic similarity = cosine proximity.
                  The pipeline is identical; only the scoring function changes.
                </p>
              </div>

              <div style={{ marginTop: 20 }}>
                <div style={s.sideLabel}>TRY THESE QUESTIONS</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 8 }}>
                  {SAMPLE_QUESTIONS.map(q => (
                    <button key={q} onClick={() => { setQuery(q); runRAG(q); }} style={s.suggBtn}>
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ── TAB: CHUNKS ── */}
          <div style={{ ...s.tabContent, display: activeTab === 'chunks' ? 'block' : 'none' }}>
            {allChunks.length === 0 ? (
              <Empty>Run a query to see how documents are split into chunks.</Empty>
            ) : (
              <>
                <div style={s.meta}>
                  {allChunks.length} chunks · {chunkSize}-word windows · ~{Math.floor(chunkSize * 0.15)}-word overlap
                  {scored.length > 0 && <> · <span style={{ color: '#f59e0b' }}>{topK} highlighted</span> were sent to LLM</>}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {allChunks.map(chunk => {
                    const color = docColorMap[chunk.docId] || '#f59e0b';
                    const isTop = scored.slice(0, topK).some(s => s.id === chunk.id);
                    return (
                      <div key={chunk.id} style={{
                        ...s.chunkCard,
                        borderLeftColor: color,
                        borderColor: isTop ? color + '44' : '#ffffff0a',
                        background: isTop ? color + '0d' : '#ffffff03',
                      }}>
                        <div style={s.chunkMeta}>
                          <span style={{ color }}>{chunk.docTitle}</span>
                          <span style={{ opacity: 0.35 }}>chunk {chunk.chunkIndex + 1}</span>
                          {isTop && <span style={{ ...s.badge, background: color + '22', color, borderColor: color + '44' }}>
                            ▲ top-{topK}
                          </span>}
                        </div>
                        <div style={s.chunkText}>{chunk.text}</div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* ── TAB: RETRIEVAL ── */}
          <div style={{ ...s.tabContent, display: activeTab === 'retrieval' ? 'block' : 'none' }}>
            {scored.length === 0 ? (
              <Empty>Run a query to see TF-IDF retrieval scores for every chunk.</Empty>
            ) : (
              <>
                <div style={s.meta}>
                  Query: <em style={{ color: '#f59e0b' }}>{currentQuery}</em>
                  &nbsp;·&nbsp;scored {scored.length} chunks &nbsp;·&nbsp; sending top-{topK} to Claude
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {scored.map((s_item, i) => {
                    const color = docColorMap[s_item.docId] || '#f59e0b';
                    const pct = (s_item.score / maxScore) * 100;
                    const selected = i < topK;
                    return (
                      <div key={s_item.id} style={{
                        ...s.scoreCard,
                        opacity: selected ? 1 : 0.45,
                        borderColor: selected ? '#ffffff12' : '#ffffff06',
                        background: selected ? '#ffffff06' : 'transparent',
                      }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                          <span style={{ fontFamily: 'monospace', fontSize: 13, color: selected ? '#f59e0b' : '#ffffff33', width: 24, flexShrink: 0 }}>
                            #{i + 1}
                          </span>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                              <span style={{ fontSize: 11, fontFamily: 'monospace', color }}>{s_item.docTitle} · chunk {s_item.chunkIndex + 1}</span>
                              {selected && <span style={{ ...s.badge, background: '#f59e0b22', color: '#f59e0b', borderColor: '#f59e0b44' }}>SELECTED</span>}
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div style={{ flex: 1, height: 5, background: '#ffffff0d', borderRadius: 3, overflow: 'hidden' }}>
                                <div style={{ height: '100%', width: pct + '%', background: selected ? color : '#ffffff22', borderRadius: 3, transition: 'width 0.5s ease' }} />
                              </div>
                              <span style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.5, flexShrink: 0 }}>{s_item.score.toFixed(4)}</span>
                            </div>
                          </div>
                        </div>
                        <div style={{ fontSize: 12, lineHeight: 1.65, opacity: 0.65, marginLeft: 34 }}>
                          {s_item.text.slice(0, 220)}{s_item.text.length > 220 ? '…' : ''}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>

          {/* ── TAB: ANSWER ── */}
          <div style={{ ...s.tabContent, display: activeTab === 'answer' ? 'block' : 'none' }}>
            {!answer && stage !== 'generating' && stage !== 'done' ? (
              <Empty>Run a query to get an AI-generated answer grounded in your documents.</Empty>
            ) : stage === 'generating' ? (
              <div style={{ fontFamily: 'monospace', fontSize: 13, color: '#f59e0b' }}>⟳ Generating answer…</div>
            ) : (
              <div style={{ maxWidth: 680 }}>
                <div style={s.sideLabel}>QUERY</div>
                <div style={s.queryBox}>{currentQuery}</div>
                <div style={{ ...s.sideLabel, marginTop: 20 }}>
                  ANSWER <span style={{ opacity: 0.4, fontWeight: 400 }}>grounded in top-{topK} retrieved chunks</span>
                </div>
                {error ? (
                  <div style={{ color: '#f87171', fontFamily: 'monospace', fontSize: 12, marginTop: 8 }}>Error: {error}</div>
                ) : (
                  <div style={s.answerBox}>{answer}</div>
                )}

                <div style={{ ...s.callout, marginTop: 20 }}>
                  <div style={s.calloutLabel}>Why RAG answers are trustworthy</div>
                  <p style={{ margin: 0, ...s.body }}>
                    The LLM is instructed to answer <em>only from the provided context</em>. If the answer
                    isn't in your top-{topK} chunks, it will say so — no hallucination. This is the grounding guarantee.
                    Check the <strong>RAG Prompt</strong> tab to see exactly what was sent.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* ── TAB: RAG PROMPT ── */}
          <div style={{ ...s.tabContent, display: activeTab === 'prompt' ? 'block' : 'none' }}>
            {!ragPrompt ? (
              <Empty>Run a query to see the exact prompt constructed and sent to Claude.</Empty>
            ) : (
              <>
                <div style={s.meta}>This is the complete prompt — context + question — sent to Claude in a single API call.</div>
                <pre style={s.promptBox}>{ragPrompt}</pre>
                <div style={s.callout}>
                  <div style={s.calloutLabel}>💡 Key Insight: The LLM only sees this</div>
                  <p style={{ margin: 0, ...s.body }}>
                    The model never "knows" your documents ahead of time. Every query rebuilds this prompt from scratch.
                    The quality of your answers is bounded by retrieval quality — if the right chunk isn't in the top-K,
                    the model can't use it. <strong>This is why retrieval is the hardest part of RAG.</strong>
                  </p>
                </div>
              </>
            )}
          </div>
        </main>
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
        input:focus, textarea:focus { outline: 1px solid #f59e0b55 !important; }
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #ffffff15; border-radius: 3px; }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}

// ─── HELPERS ───────────────────────────────────────────────────────────────────

const sleep = ms => new Promise(r => setTimeout(r, ms));

function Empty({ children }) {
  return (
    <div style={{ fontFamily: 'monospace', fontSize: 13, color: '#ffffff33', paddingTop: 8 }}>
      {children}
    </div>
  );
}

// ─── STYLES ────────────────────────────────────────────────────────────────────

const s = {
  root: {
    minHeight: '100vh', background: '#08080e', color: '#ddd6c8',
    fontFamily: "'Palatino Linotype', Palatino, Georgia, serif",
    display: 'flex', flexDirection: 'column', fontSize: 14,
  },
  header: {
    borderBottom: '1px solid #16161f', padding: '14px 28px',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0,
  },
  logo: {
    fontFamily: "'Courier New', monospace", fontSize: 18, color: '#f59e0b',
    letterSpacing: '0.06em', lineHeight: 1,
  },
  subtitle: { fontSize: 11, fontFamily: 'monospace', opacity: 0.35, marginTop: 4 },
  pill: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '3px 10px', borderRadius: 20, border: '1px solid',
    fontSize: 11, fontFamily: 'monospace', transition: 'all 0.3s',
  },
  dot: {
    width: 6, height: 6, borderRadius: '50%', background: '#f59e0b',
    animation: 'blink 0.8s infinite', flexShrink: 0,
  },
  layout: { display: 'flex', flex: 1, overflow: 'hidden' },
  sidebar: {
    width: 264, borderRight: '1px solid #16161f',
    padding: '18px 16px', flexShrink: 0, overflowY: 'auto',
  },
  sideSection: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  sideLabel: {
    fontSize: 10, fontFamily: 'monospace', color: '#f59e0b',
    letterSpacing: '0.12em', marginBottom: 6,
  },
  addBtn: {
    background: 'transparent', border: '1px solid #f59e0b44', color: '#f59e0b',
    padding: '3px 9px', borderRadius: 4, cursor: 'pointer', fontSize: 10, fontFamily: 'monospace',
  },
  addForm: {
    background: '#ffffff06', borderRadius: 8, padding: 12,
    border: '1px solid #ffffff0d', marginBottom: 12,
    display: 'flex', flexDirection: 'column', gap: 8,
  },
  input: {
    width: '100%', background: '#ffffff08', border: '1px solid #ffffff10',
    borderRadius: 4, padding: '6px 8px', color: '#ddd6c8', fontSize: 12,
    fontFamily: 'Palatino Linotype, Palatino, Georgia, serif',
  },
  ingestBtn: {
    width: '100%', background: '#f59e0b', border: 'none', borderRadius: 4,
    padding: '7px', color: '#08080e', cursor: 'pointer', fontSize: 11,
    fontFamily: 'monospace', fontWeight: 'bold', letterSpacing: '0.06em',
  },
  docCard: {
    padding: '10px 12px', borderRadius: 6, border: '1px solid',
    borderLeft: '3px solid',
  },
  configLabel: {
    fontSize: 11, fontFamily: 'monospace', opacity: 0.55, display: 'block', marginBottom: 4,
  },
  range: { width: '100%', accentColor: '#f59e0b', marginBottom: 12 },
  main: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  queryBar: {
    padding: '14px 22px', borderBottom: '1px solid #16161f',
    display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0,
  },
  queryArrow: {
    position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)',
    fontFamily: 'monospace', fontSize: 16, color: '#f59e0b', opacity: 0.6,
  },
  queryInput: {
    width: '100%', background: '#ffffff08', border: '1px solid #ffffff12',
    borderRadius: 6, padding: '9px 14px 9px 28px', color: '#ddd6c8',
    fontSize: 14, fontFamily: 'Palatino Linotype, Palatino, Georgia, serif',
  },
  runBtn: {
    background: '#f59e0b', border: 'none', borderRadius: 6,
    padding: '9px 18px', color: '#08080e', cursor: 'pointer',
    fontSize: 12, fontFamily: 'monospace', fontWeight: 'bold',
    letterSpacing: '0.05em', flexShrink: 0,
  },
  tabs: {
    display: 'flex', borderBottom: '1px solid #16161f',
    padding: '0 22px', flexShrink: 0, overflowX: 'auto',
  },
  tabBtn: {
    background: 'transparent', border: 'none', borderBottom: '2px solid transparent',
    padding: '11px 14px', cursor: 'pointer', fontSize: 11,
    fontFamily: 'monospace', letterSpacing: '0.04em', flexShrink: 0,
  },
  tabContent: { flex: 1, overflowY: 'auto', padding: '22px 24px' },
  h2: { color: '#f59e0b', fontFamily: 'monospace', fontSize: 15, letterSpacing: '0.06em', marginTop: 0, marginBottom: 10 },
  body: { lineHeight: 1.8, opacity: 0.82, marginBottom: 12, fontSize: 13 },
  stepCard: {
    display: 'flex', gap: 14, alignItems: 'flex-start',
    padding: '12px 14px', borderRadius: 7,
    background: '#ffffff05', border: '1px solid #ffffff0a',
  },
  stepNum: { fontFamily: 'monospace', fontSize: 18, color: '#f59e0b', opacity: 0.55, flexShrink: 0, lineHeight: 1.2 },
  stepTitle: { fontFamily: 'monospace', fontSize: 12, color: '#f59e0b', marginBottom: 3, letterSpacing: '0.05em' },
  callout: { padding: 16, borderRadius: 8, background: '#f59e0b0d', border: '1px solid #f59e0b2a' },
  calloutLabel: { fontFamily: 'monospace', fontSize: 11, color: '#f59e0b', letterSpacing: '0.08em', marginBottom: 8 },
  suggBtn: {
    background: 'transparent', border: '1px solid #ffffff10', color: '#ddd6c8',
    borderRadius: 5, padding: '7px 12px', cursor: 'pointer', fontSize: 12,
    fontFamily: 'Palatino Linotype, Palatino, Georgia, serif', textAlign: 'left',
    transition: 'border-color 0.2s, background 0.2s',
  },
  meta: { fontSize: 11, fontFamily: 'monospace', opacity: 0.45, marginBottom: 14, lineHeight: 1.6 },
  chunkCard: {
    padding: '10px 13px', borderRadius: 6,
    border: '1px solid', borderLeft: '3px solid',
  },
  chunkMeta: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 10, fontFamily: 'monospace', marginBottom: 5, opacity: 0.7 },
  chunkText: { fontSize: 12, lineHeight: 1.7, opacity: 0.78 },
  badge: {
    fontSize: 9, fontFamily: 'monospace', padding: '1px 6px',
    borderRadius: 3, border: '1px solid', letterSpacing: '0.05em',
  },
  scoreCard: { padding: '12px 14px', borderRadius: 7, border: '1px solid' },
  queryBox: {
    padding: '11px 14px', borderRadius: 6, background: '#ffffff08',
    border: '1px solid #ffffff10', fontSize: 14, lineHeight: 1.6, marginTop: 6,
  },
  answerBox: {
    padding: '16px 18px', borderRadius: 8, background: '#ffffff06',
    border: '1px solid #f59e0b1e', fontSize: 14, lineHeight: 1.85,
    whiteSpace: 'pre-wrap', marginTop: 6,
  },
  promptBox: {
    background: '#0a0a12', border: '1px solid #16161f', borderRadius: 8,
    padding: '16px 18px', fontSize: 11, lineHeight: 1.75, overflowX: 'auto',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#b0a898',
    fontFamily: "'Courier New', monospace", marginTop: 10,
  },
};
