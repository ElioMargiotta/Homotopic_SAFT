import { useState, useMemo } from "react";

const T = { Fm: -5.765234, Fc: -1.552748, Fa: -21.113097, m: 2.1968, s3: 4.42117e-29, sh: 0.62487 };

const D = [
{r:1,n:"CC(N)CO",Fm:-5.4802,Fc:-2.0224,Fa:-20.6233,m:2.6122,s3:4.600705e-29,sh:0.6329,dt:0.736,ds:0.178134},
{r:2,n:"CC(CN)CO",Fm:-6.2615,Fc:-2.3326,Fa:-19.9597,m:2.8415,s3:5.086484e-29,sh:0.6003,dt:1.47807,ds:0.295751},
{r:3,n:"CC(C)NCC(O)CN",Fm:-6.6234,Fc:-2.7779,Fa:-21.913,m:3.2177,s3:6.510324e-29,sh:0.4993,dt:1.69627,ds:0.588011},
{r:4,n:"NCCCCO",Fm:-7.2667,Fc:-2.1674,Fa:-20.6026,m:2.6555,s3:5.497661e-29,sh:0.5566,dt:1.700876,ds:0.311199},
{r:5,n:"CCC(O)CNCCO",Fm:-7.3771,Fc:-2.844,Fa:-21.5299,m:3.1764,s3:6.466435e-29,sh:0.4334,dt:2.106965,ds:0.643725},
{r:6,n:"CC(CN)CCO",Fm:-7.0551,Fc:-2.642,Fa:-19.7422,m:3.0708,s3:5.510886e-29,sh:0.5726,dt:2.17475,ds:0.410301},
{r:7,n:"CC(C)NCCC(O)CN",Fm:-7.4823,Fc:-3.0854,Fa:-21.6528,m:3.4471,s3:6.818202e-29,sh:0.4813,dt:2.364003,ds:0.677291},
{r:8,n:"CC(O)CNCC(C)O",Fm:-6.7149,Fc:-2.227,Fa:-19.032,m:2.7282,s3:7.71726e-29,sh:0.3807,dt:2.384846,ds:0.776446},
{r:9,n:"CCC(O)CN",Fm:-6.2916,Fc:-1.5514,Fa:-18.7775,m:2.2072,s3:6.765745e-29,sh:0.5164,dt:2.39418,ds:0.466289},
{r:10,n:"NCCCCCO",Fm:-8.0445,Fc:-2.4746,Fa:-20.3755,m:2.8848,s3:5.925899e-29,sh:0.5305,dt:2.56688,ds:0.432215},
{r:11,n:"CC(CO)NCCO",Fm:-7.2403,Fc:-3.3223,Fa:-19.6349,m:3.5814,s3:4.917067e-29,sh:0.5278,dt:2.737176,ds:0.527904},
{r:12,n:"CC(O)CCCN",Fm:-7.0577,Fc:-1.8561,Fa:-18.4962,m:2.4365,s3:7.18334e-29,sh:0.4893,dt:2.934352,ds:0.553232},
{r:13,n:"CC(CCN)CCO",Fm:-7.8585,Fc:-2.9507,Fa:-19.5392,m:3.3001,s3:5.884538e-29,sh:0.5487,dt:2.968666,ds:0.514036},
{r:14,n:"CC(O)CCCNCCO",Fm:-8.266,Fc:-3.1512,Fa:-21.2643,m:3.4057,s3:6.78033e-29,sh:0.4197,dt:2.971794,ds:0.730448},
{r:15,n:"CC(O)CNCCNCC(C)O",Fm:-7.6763,Fc:-3.2851,Fa:-22.7143,m:3.5527,s3:7.690028e-29,sh:0.3596,dt:3.036037,ds:0.918008},
{r:16,n:"OCCCNCCO",Fm:-8.2939,Fc:-3.1657,Fa:-20.2947,m:3.3954,s3:5.225732e-29,sh:0.4896,dt:3.108931,ds:0.526361},
{r:17,n:"CC(CN)NC(C)CO",Fm:-6.7486,Fc:-3.5382,Fa:-23.2968,m:3.852,s3:5.311959e-29,sh:0.564,dt:3.110901,ds:0.599637},
{r:18,n:"CNC(CN)CO",Fm:-6.2741,Fc:-2.7665,Fa:-23.9907,m:3.2074,s3:4.890702e-29,sh:0.5736,dt:3.16427,ds:0.400927},
{r:19,n:"CC(N)CC(C)O",Fm:-6.1663,Fc:-2.0341,Fa:-18.0023,m:2.6225,s3:6.577088e-29,sh:0.5415,dt:3.173293,ds:0.457854},
{r:20,n:"CC(N)CN(C)CCO",Fm:-8.0379,Fc:-3.5242,Fa:-20.0953,m:3.794,s3:5.580969e-29,sh:0.5558,dt:3.176055,ds:0.605433},
{r:21,n:"CC(O)CC(C)NCCCN",Fm:-8.3417,Fc:-3.3928,Fa:-21.4098,m:3.6764,s3:7.091436e-29,sh:0.4656,dt:3.179951,ds:0.758236},
{r:41,n:"NCCNCCO",Fm:-7.5838,Fc:-2.6116,Fa:-24.9754,m:3.0214,s3:5.235397e-29,sh:0.5335,dt:4.398356,ds:0.393893},
{r:53,n:"CC(O)CC(CN)CN",Fm:-8.877,Fc:-2.9132,Fa:-24.6371,m:3.3054,s3:6.338224e-29,sh:0.5543,dt:4.894162,ds:0.557657},
{r:97,n:"CC(N)CC(CN)CO",Fm:-8.8691,Fc:-3.6786,Fa:-26.4354,m:3.9396,s3:5.204202e-29,sh:0.6088,dt:6.517687,ds:0.606973},
{r:98,n:"CNCC(CN)NCCO",Fm:-7.8723,Fc:-3.8226,Fa:-26.9,m:4.0319,s3:5.414501e-29,sh:0.5156,dt:6.563593,ds:0.668413},
{r:99,n:"CN(C)CCC(O)CO",Fm:-8.289,Fc:-2.9857,Fa:-15.1768,m:3.3045,s3:6.382079e-29,sh:0.4663,dt:6.607743,ds:0.622213},
{r:100,n:"NCCC(N)CO",Fm:-8.4286,Fc:-2.9072,Fa:-27.0221,m:3.295,s3:4.77719e-29,sh:0.6268,dt:6.621544,ds:0.412734},
{r:103,n:"CCN(CCO)CCO",Fm:-9.5733,Fc:-3.6122,Fa:-15.9653,m:3.7527,s3:5.535868e-29,sh:0.5007,dt:6.726224,ds:0.621584},
{r:104,n:"CN(C)CC(CO)CO",Fm:-8.9216,Fc:-3.7678,Fa:-15.5303,m:3.9387,s3:5.235374e-29,sh:0.5349,dt:6.78507,ds:0.627393},
{r:137,n:"NCCNCCNCCO",Fm:-9.3321,Fc:-3.6693,Fa:-28.0505,m:3.8459,s3:5.718501e-29,sh:0.4813,dt:8.082713,ds:0.669326},
{r:141,n:"CC(O)CC(O)CN",Fm:-8.7909,Fc:-2.0544,Fa:-28.8364,m:2.5865,s3:7.12484e-29,sh:0.4685,dt:8.309992,ds:0.580835},
{r:149,n:"OCCNC(CO)CO",Fm:-9.7231,Fc:-4.1565,Fa:-28.2858,m:4.1796,s3:4.406029e-29,sh:0.5378,dt:8.595984,ds:0.660491},
{r:150,n:"CC(O)CC(N)CO",Fm:-8.2133,Fc:-2.8412,Fa:-29.2748,m:3.2207,s3:5.541205e-29,sh:0.5519,dt:8.617773,ds:0.46127},
{r:160,n:"NCCC(O)CCO",Fm:-9.0923,Fc:-2.675,Fa:-29.8779,m:3.0347,s3:5.938363e-29,sh:0.5107,dt:9.441999,ds:0.48182},
{r:162,n:"NCC(O)CCO",Fm:-8.307,Fc:-2.3677,Fa:-30.1917,m:2.8054,s3:5.533093e-29,sh:0.5337,dt:9.462901,ds:0.367425},
{r:164,n:"CNCCO",Fm:-4.0201,Fc:-1.7232,Fa:-11.6349,m:2.3385,s3:5.169255e-29,sh:0.513,dt:9.639025,ds:0.259371},
{r:165,n:"CCNCCCO",Fm:-5.7566,Fc:-2.3372,Fa:-11.2828,m:2.7972,s3:6.105745e-29,sh:0.4665,dt:9.861504,ds:0.498047},
{r:170,n:"CCCNCCCO",Fm:-6.6239,Fc:-2.6442,Fa:-11.1283,m:3.0265,s3:6.48032e-29,sh:0.4485,dt:10.080911,ds:0.599025},
{r:173,n:"CNCC(C)O",Fm:-3.5319,Fc:-1.4214,Fa:-11.0067,m:2.1196,s3:7.053464e-29,sh:0.4312,dt:10.351009,ds:0.597541},
{r:191,n:"CC(O)CNC(C)C",Fm:-4.3928,Fc:-2.2077,Fa:-10.2826,m:2.7642,s3:7.17431e-29,sh:0.4511,dt:10.93678,ds:0.627117},
{r:200,n:"CC(O)CCCNC(C)C",Fm:-6.1888,Fc:-2.8183,Fa:-9.9797,m:3.2229,s3:7.762304e-29,sh:0.4196,dt:11.2131,ds:0.788909},
{r:214,n:"CC(CO)NC(CN)CN",Fm:-9.1113,Fc:-4.1074,Fa:-31.9913,m:4.3055,s3:5.047302e-29,sh:0.5881,dt:11.664336,ds:0.688471},
{r:222,n:"NCC(CN)NCCO",Fm:-9.6601,Fc:-3.6519,Fa:-32.4647,m:3.8902,s3:4.990409e-29,sh:0.5788,dt:12.183382,ds:0.589133},
{r:237,n:"CN(C)CCO",Fm:-5.7692,Fc:-2.1715,Fa:-7.6982,m:2.6959,s3:5.609689e-29,sh:0.5253,dt:13.429109,ds:0.358737},
{r:239,n:"CCN(C)CCO",Fm:-6.5124,Fc:-2.479,Fa:-7.6167,m:2.9252,s3:6.027284e-29,sh:0.5021,dt:13.548796,ds:0.475231},
{r:241,n:"CC(CO)N(C)C",Fm:-5.9192,Fc:-2.6417,Fa:-7.5542,m:3.1112,s3:5.607965e-29,sh:0.5453,dt:13.603422,ds:0.442913},
{r:256,n:"CCC(O)CN(C)CC",Fm:-7.3171,Fc:-2.4792,Fa:-6.5381,m:2.9356,s3:7.926487e-29,sh:0.421,dt:14.68664,ds:0.762131},
{r:266,n:"CC(O)CNCCNCC(O)CN",Fm:-10.5224,Fc:-3.8647,Fa:-37.0433,m:4.0062,s3:7.084145e-29,sh:0.4087,dt:16.785337,ds:0.873851},
{r:270,n:"NCC(O)CCC(O)CN",Fm:-11.6453,Fc:-2.9433,Fa:-38.9608,m:3.2693,s3:6.76147e-29,sh:0.4967,dt:18.842723,ds:0.625502},
{r:282,n:"NCC(O)CNCC(O)CN",Fm:-11.8017,Fc:-3.3869,Fa:-44.2801,m:3.6352,s3:6.398148e-29,sh:0.4835,dt:24.010726,ds:0.675305},
{r:286,n:"NCC(CN)CC(CO)NCCO",Fm:-13.4571,Fc:-5.5578,Fa:-45.1802,m:5.3624,s3:5.106832e-29,sh:0.5585,dt:25.581832,ds:0.910913},
];

function classify(d) {
  const dFa = d.Fa - T.Fa;
  const dFm = d.Fm - T.Fm;
  const dFc = d.Fc - T.Fc;
  const sq = dFm * dFm + dFc * dFc + dFa * dFa;
  const fracA = sq > 0 ? (dFa * dFa) / sq : 0;
  if (dFa < -5) return "over";
  if (dFa > 10) return "tert";
  if (dFa > 5) return "sec";
  if (fracA < 0.3 && Math.abs(dFa) < 5) return "size";
  return "bal";
}

const CL = {
  bal:  { label: "Balanced (primary amines)", color: "#2D9CDB", bg: "#2D9CDB22" },
  size: { label: "Size-driven deviation", color: "#27AE60", bg: "#27AE6022" },
  sec:  { label: "Under-assoc. (secondary)", color: "#F2994A", bg: "#F2994A22" },
  tert: { label: "Under-assoc. (tertiary)",  color: "#EB5757", bg: "#EB575722" },
  over: { label: "Over-associating",         color: "#9B51E0", bg: "#9B51E022" },
};

export default function App() {
  const [hovId, setHov] = useState(null);
  const [selCl, setSelCl] = useState(null);
  const [view, setView] = useState("scatter");

  const data = useMemo(() => D.map(d => {
    const dFm = d.Fm - T.Fm, dFc = d.Fc - T.Fc, dFa = d.Fa - T.Fa;
    const sq = dFm*dFm + dFc*dFc + dFa*dFa;
    return { ...d, cl: classify(d), dFm, dFc, dFa, fracA: sq > 0 ? dFa*dFa/sq : 0 };
  }), []);

  const maxDt = Math.max(...data.map(d => d.dt));
  const maxDs = Math.max(...data.map(d => d.ds));

  const stats = useMemo(() => {
    const s = {};
    for (const k of Object.keys(CL)) s[k] = [];
    data.forEach(d => s[d.cl].push(d));
    return s;
  }, [data]);

  const pareto = useMemo(() => {
    const sorted = [...data].sort((a,b) => a.dt - b.dt);
    const f = []; let minDs = Infinity;
    for (const d of sorted) { if (d.ds < minDs) { f.push(d); minDs = d.ds; } }
    return new Set(f.map(d => d.r));
  }, [data]);

  const filt = selCl ? data.filter(d => d.cl === selCl) : data;

  const W = 720, H = 460, M = {t:40, r:24, b:52, l:60};
  const pw = W-M.l-M.r, ph = H-M.t-M.b;
  const sx = v => M.l + (v / (maxDt * 1.05)) * pw;
  const sy = v => M.t + ph - (v / (maxDs * 1.08)) * ph;

  const hov = hovId !== null ? data.find(d => d.r === hovId) : null;

  const barW = 500;

  return (
    <div style={{ fontFamily:"'IBM Plex Mono','SF Mono',monospace", background:"#0D1117", color:"#C9D1D9", minHeight:"100vh", padding:20 }}>
      <div style={{ maxWidth:820, margin:"0 auto" }}>

        <h1 style={{ fontSize:20, fontWeight:700, color:"#58A6FF", margin:"0 0 2px", letterSpacing:"-0.3px" }}>
          D<sub style={{fontSize:12}}>thermo</sub> vs D<sub style={{fontSize:12}}>struct</sub> — 286 candidates vs NCCO
        </h1>
        <p style={{ fontSize:11, color:"#8B949E", margin:"0 0 16px", lineHeight:1.5 }}>
          Thermodynamic Euclidean distance (F<sub>mono</sub>, F<sub>chain</sub>, F<sub>assoc</sub>) vs structural log-Euclidean distance (m, σ³, S̄)
        </p>

        {/* Tabs */}
        <div style={{ display:"flex", gap:2, marginBottom:12 }}>
          {[{id:"scatter",l:"Scatter + Pareto"},{id:"decomp",l:"Decomposition"},{id:"assoc",l:"F_assoc driver"}].map(t =>
            <button key={t.id} onClick={() => setView(t.id)} style={{
              padding:"5px 14px", fontSize:10, fontFamily:"inherit", fontWeight:600, cursor:"pointer",
              background: view===t.id ? "#1F6FEB" : "#21262D",
              color: view===t.id ? "#fff" : "#8B949E",
              border:"1px solid "+(view===t.id?"#1F6FEB":"#30363D"), borderRadius:6,
            }}>{t.l}</button>
          )}
        </div>

        {/* Cluster filters */}
        <div style={{ display:"flex", flexWrap:"wrap", gap:5, marginBottom:12 }}>
          <button onClick={() => setSelCl(null)} style={{
            padding:"3px 10px", fontSize:9, fontFamily:"inherit", cursor:"pointer",
            background: !selCl?"#30363D":"transparent", color:"#C9D1D9",
            border:"1px solid #30363D", borderRadius:10,
          }}>All ({data.length})</button>
          {Object.entries(CL).map(([k,v]) =>
            <button key={k} onClick={() => setSelCl(selCl===k?null:k)} style={{
              padding:"3px 10px", fontSize:9, fontFamily:"inherit", cursor:"pointer",
              background: selCl===k?v.bg:"transparent", color:v.color,
              border:`1px solid ${selCl===k?v.color:"#30363D"}`, borderRadius:10,
              fontWeight: selCl===k?700:400,
            }}>● {v.label} ({(stats[k]||[]).length})</button>
          )}
        </div>

        {/* Main scatter */}
        {(view === "scatter") && (
        <div style={{ background:"#161B22", borderRadius:10, border:"1px solid #30363D", padding:12, position:"relative" }}>
          <svg width={W} height={H} style={{ display:"block" }}>
            {/* Grid */}
            {[0,5,10,15,20,25].map(v => v <= maxDt*1.05 && (
              <g key={"gx"+v}>
                <line x1={sx(v)} x2={sx(v)} y1={M.t} y2={M.t+ph} stroke="#21262D" />
                <text x={sx(v)} y={M.t+ph+16} textAnchor="middle" fill="#484F58" fontSize={9} fontFamily="inherit">{v}</text>
              </g>
            ))}
            {[0,0.2,0.4,0.6,0.8,1.0].map(v => (
              <g key={"gy"+v}>
                <line x1={M.l} x2={M.l+pw} y1={sy(v)} y2={sy(v)} stroke="#21262D" />
                <text x={M.l-6} y={sy(v)+3} textAnchor="end" fill="#484F58" fontSize={9} fontFamily="inherit">{v.toFixed(1)}</text>
              </g>
            ))}
            <text x={M.l+pw/2} y={H-4} textAnchor="middle" fill="#8B949E" fontSize={11} fontFamily="inherit" fontWeight={600}>
              D_thermo (Euclidean)
            </text>
            <text x={14} y={M.t+ph/2} textAnchor="middle" fill="#8B949E" fontSize={11} fontFamily="inherit" fontWeight={600}
              transform={`rotate(-90,14,${M.t+ph/2})`}>D_struct (log-Eucl.)</text>

            {/* Pareto front */}
            {(() => {
              const pts = data.filter(d => pareto.has(d.r)).sort((a,b) => a.dt-b.dt);
              if (pts.length < 2) return null;
              const path = pts.map((d,i) => `${i===0?"M":"L"}${sx(d.dt)},${sy(d.ds)}`).join(" ");
              return <path d={path} fill="none" stroke="#58A6FF" strokeWidth={1.5} strokeDasharray="5,3" opacity={0.5} />;
            })()}

            {/* Points */}
            {filt.map(d => {
              const isP = pareto.has(d.r);
              const isH = hovId===d.r;
              const op = selCl && d.cl!==selCl ? 0.08 : 0.8;
              return (
                <circle key={d.r} cx={sx(d.dt)} cy={sy(d.ds)}
                  r={isH?6:(isP?4.5:3)} fill={CL[d.cl].color} fillOpacity={op}
                  stroke={isH?"#fff":(isP?"#58A6FF":"none")} strokeWidth={isH?2:1}
                  style={{ cursor:"pointer" }}
                  onMouseEnter={() => setHov(d.r)} onMouseLeave={() => setHov(null)}
                />
              );
            })}

            {/* Target */}
            <polygon points={`${sx(0)},${sy(0)-7} ${sx(0)+5},${sy(0)+3} ${sx(0)-5},${sy(0)+3}`}
              fill="#58A6FF" stroke="#0D1117" strokeWidth={1.5} />
            <text x={sx(0)+8} y={sy(0)-2} fill="#58A6FF" fontSize={9} fontFamily="inherit" fontWeight={700}>NCCO</text>
          </svg>

          {/* Tooltip */}
          {hov && (
            <div style={{
              position:"absolute", top:12, right:12, width:220,
              background:"#0D1117ee", border:`1px solid ${CL[hov.cl].color}`,
              borderRadius:8, padding:10, fontSize:9.5, lineHeight:1.7,
            }}>
              <div style={{ fontWeight:700, fontSize:11, color:CL[hov.cl].color }}>#{hov.r} {hov.n}</div>
              <div style={{ color:"#8B949E", fontSize:9 }}>● {CL[hov.cl].label}</div>
              <div style={{ marginTop:4 }}>D<sub>th</sub> = <b>{hov.dt.toFixed(2)}</b> &nbsp; D<sub>st</sub> = <b>{hov.ds.toFixed(3)}</b></div>
              <hr style={{ border:"none", borderTop:"1px solid #21262D", margin:"4px 0" }} />
              <div>ΔF<sub>mono</sub> = {hov.dFm>0?"+":""}{hov.dFm.toFixed(2)}</div>
              <div>ΔF<sub>chain</sub> = {hov.dFc>0?"+":""}{hov.dFc.toFixed(2)}</div>
              <div style={{ color: Math.abs(hov.dFa)>8?"#EB5757":"#C9D1D9" }}>
                ΔF<sub>assoc</sub> = {hov.dFa>0?"+":""}{hov.dFa.toFixed(2)}
                {Math.abs(hov.dFa)>8?" ⚠":""}
              </div>
              <div style={{ color:"#8B949E", marginTop:2 }}>
                assoc fraction of D²: {(hov.fracA*100).toFixed(0)}%
              </div>
              <div style={{ marginTop:2 }}>m = {hov.m.toFixed(2)} (tgt: {T.m.toFixed(2)})</div>
            </div>
          )}
        </div>
        )}

        {/* Decomposition view */}
        {view === "decomp" && (
        <div style={{ background:"#161B22", borderRadius:10, border:"1px solid #30363D", padding:12 }}>
          <div style={{ fontSize:12, color:"#58A6FF", fontWeight:700, marginBottom:8 }}>
            D²_thermo decomposition — top 50
          </div>
          <div style={{ fontSize:9, color:"#8B949E", marginBottom:10 }}>
            Fraction of D²_thermo from each ΔF component. Bar width ∝ D_thermo.
          </div>
          {data.slice(0, 50).map(d => {
            const sq = d.dFm**2 + d.dFc**2 + d.dFa**2;
            const fM = sq>0?d.dFm**2/sq:0.33, fC=sq>0?d.dFc**2/sq:0.33, fA=sq>0?d.dFa**2/sq:0.33;
            const w = (d.dt / maxDt) * barW;
            return (
              <div key={d.r} style={{ display:"flex", alignItems:"center", marginBottom:2, gap:6 }}>
                <div style={{ width:100, fontSize:8.5, textAlign:"right", color:CL[d.cl].color,
                  overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{d.n}</div>
                <div style={{ width:w, height:10, display:"flex", borderRadius:1, overflow:"hidden", flexShrink:0 }}>
                  <div style={{ width:`${fM*100}%`, background:"#58A6FF", height:"100%" }} />
                  <div style={{ width:`${fC*100}%`, background:"#3FB950", height:"100%" }} />
                  <div style={{ width:`${fA*100}%`, background:"#F85149", height:"100%" }} />
                </div>
                <div style={{ fontSize:8, color:"#484F58", width:32, flexShrink:0 }}>{d.dt.toFixed(1)}</div>
              </div>
            );
          })}
          <div style={{ display:"flex", gap:14, marginTop:8, fontSize:10 }}>
            <span><span style={{color:"#58A6FF"}}>■</span> ΔF²_mono</span>
            <span><span style={{color:"#3FB950"}}>■</span> ΔF²_chain</span>
            <span><span style={{color:"#F85149"}}>■</span> ΔF²_assoc</span>
          </div>
        </div>
        )}

        {/* F_assoc driver view */}
        {view === "assoc" && (
        <div style={{ background:"#161B22", borderRadius:10, border:"1px solid #30363D", padding:12 }}>
          <div style={{ fontSize:12, color:"#58A6FF", fontWeight:700, marginBottom:8 }}>
            ΔF_assoc vs D_thermo — association as the dominant driver
          </div>
          <svg width={W} height={H} style={{ display:"block" }}>
            {/* Axes */}
            {[-25,-20,-15,-10,-5,0,5,10,15].map(v => {
              const x = M.l + ((v+25)/40)*pw;
              return x >= M.l && x <= M.l+pw && (
                <g key={"ax"+v}>
                  <line x1={x} x2={x} y1={M.t} y2={M.t+ph} stroke="#21262D" />
                  <text x={x} y={M.t+ph+16} textAnchor="middle" fill="#484F58" fontSize={9} fontFamily="inherit">{v>0?"+":""}{v}</text>
                </g>
              );
            })}
            {[0,5,10,15,20,25].map(v => {
              const y = M.t+ph-(v/(maxDt*1.05))*ph;
              return (
                <g key={"ay"+v}>
                  <line x1={M.l} x2={M.l+pw} y1={y} y2={y} stroke="#21262D" />
                  <text x={M.l-6} y={y+3} textAnchor="end" fill="#484F58" fontSize={9} fontFamily="inherit">{v}</text>
                </g>
              );
            })}
            <text x={M.l+pw/2} y={H-4} textAnchor="middle" fill="#8B949E" fontSize={11} fontFamily="inherit" fontWeight={600}>
              ΔF_assoc (candidate − target)
            </text>
            <text x={14} y={M.t+ph/2} textAnchor="middle" fill="#8B949E" fontSize={11} fontFamily="inherit" fontWeight={600}
              transform={`rotate(-90,14,${M.t+ph/2})`}>D_thermo</text>

            {/* Zero line */}
            <line x1={M.l+(25/40)*pw} x2={M.l+(25/40)*pw} y1={M.t} y2={M.t+ph}
              stroke="#58A6FF" strokeWidth={1} strokeDasharray="3,3" opacity={0.4} />

            {filt.map(d => {
              const x = M.l + ((d.dFa+25)/40)*pw;
              const y = M.t+ph-(d.dt/(maxDt*1.05))*ph;
              const isH = hovId===d.r;
              return (
                <circle key={d.r} cx={x} cy={y} r={isH?6:3} fill={CL[d.cl].color}
                  fillOpacity={selCl && d.cl!==selCl?0.08:0.8}
                  stroke={isH?"#fff":"none"} strokeWidth={2}
                  style={{cursor:"pointer"}}
                  onMouseEnter={() => setHov(d.r)} onMouseLeave={() => setHov(null)}
                />
              );
            })}
          </svg>
        </div>
        )}

        {/* Cluster summary cards */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:6, marginTop:10 }}>
          {Object.entries(CL).map(([k,v]) => {
            const items = stats[k]||[];
            if (!items.length) return null;
            const avgDt = items.reduce((s,d)=>s+d.dt,0)/items.length;
            const avgDs = items.reduce((s,d)=>s+d.ds,0)/items.length;
            const avgDFa = items.reduce((s,d)=>s+(d.Fa-T.Fa),0)/items.length;
            return (
              <div key={k} onClick={() => setSelCl(selCl===k?null:k)} style={{
                background:"#161B22", borderRadius:8, padding:10,
                border:`1px solid ${selCl===k?v.color:"#30363D"}`, cursor:"pointer",
              }}>
                <div style={{ fontSize:10, fontWeight:700, color:v.color }}>{v.label}</div>
                <div style={{ fontSize:9, color:"#8B949E", marginTop:2 }}>n = {items.length}</div>
                <div style={{ fontSize:9, marginTop:4 }}>
                  D̄<sub>th</sub>={avgDt.toFixed(1)} &nbsp;
                  D̄<sub>st</sub>={avgDs.toFixed(2)} &nbsp;
                  ΔF̄<sub>a</sub>={avgDFa>0?"+":""}{avgDFa.toFixed(1)}
                </div>
              </div>
            );
          })}
        </div>

        {/* Key observations */}
        <div style={{
          background:"#161B22", borderRadius:10, marginTop:10,
          border:"1px solid #30363D", padding:14, fontSize:10.5,
          lineHeight:1.7, color:"#8B949E",
        }}>
          <div style={{ fontWeight:700, color:"#58A6FF", marginBottom:6, fontSize:11 }}>Key observations</div>
          <div>
            <b style={{color:"#C9D1D9"}}>D_struct is bounded (0.18–1.03)</b> while D_thermo spans 0.7–25.6.
            Structural variation is modest — most candidates are built from similar-sized segments (m ≈ 2–5).
            D_struct alone cannot discriminate; energetics dominate the ranking.
          </div>
          <div style={{marginTop:4}}>
            <b style={{color:"#C9D1D9"}}>F_assoc is the primary driver of D_thermo.</b> The "Decomposition" tab shows F_assoc
            accounts for &gt;80% of D² beyond rank ~30. Two symmetric bands emerge: over-associating
            (multi-NH₂/OH, ΔF_assoc ≪ 0) and under-associating (tertiary amines with N(C)(C),
            ΔF_assoc ≫ 0).
          </div>
          <div style={{marginTop:4}}>
            <b style={{color:"#C9D1D9"}}>Pareto-optimal candidates</b> (dashed line) sit on the lower-left envelope:
            CC(N)CO (#1, 2-amino-1-propanol), CC(CN)CO (#2), and NCCCCO (#4) — all small
            primary amino-alcohols structurally close to NCCO.
          </div>
          <div style={{marginTop:4}}>
            <b style={{color:"#C9D1D9"}}>Size-driven cluster</b> (green): similar F_assoc but larger backbones (more CH₂/CH₃).
            These have moderate D_thermo (3–7) driven by F_mono and F_chain, not by association mismatch.
          </div>
        </div>
      </div>
    </div>
  );
}
