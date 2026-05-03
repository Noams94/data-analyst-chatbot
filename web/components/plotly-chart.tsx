/**
 * Thin wrapper that wires react-plotly.js's factory to plotly.js-dist-min
 * (the installed package). This file is always loaded via next/dynamic with
 * ssr:false so Plotly's browser-only globals are never hit on the server.
 */
// eslint-disable-next-line @typescript-eslint/no-require-imports
const createPlotlyComponent = require("react-plotly.js/factory");
// eslint-disable-next-line @typescript-eslint/no-require-imports
const Plotly = require("plotly.js-dist-min");

const Plot = createPlotlyComponent(Plotly);
export default Plot;
