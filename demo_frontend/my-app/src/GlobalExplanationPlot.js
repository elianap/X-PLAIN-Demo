import Plot from "react-plotly.js"
import React from "react"

function getTrace(differences, names) {
  return {
    type: "bar",
    x: differences,
    y: names,
    orientation: "h",
    marker: {
      color: differences.map(x => {
        return [
          "rgb(165,0,38)",
          "rgb(215,48,39)",
          "rgb(244,109,67)",
          "rgb(253,174,97)",
          "rgb(254,224,144)",
          "rgb(255,255,191)",
          "rgb(224,243,248)",
          "rgb(171,217,233)",
          "rgb(116,173,209)",
          "rgb(69,117,180)",
          "rgb(49,54,149)"
        ][(((x + 1) / 2) * 10) | 0]
      }),
      line: {
        width: 1
      }
    }
  }
}

function getNames(globalExplanation) {
  return globalExplanation.x
}

function getDifferences(globalExplanation) {
  return globalExplanation.y
}

//    {compare ? '' : '{{}}'}
//style={{width: '100%', height: '100%'}}
function GlobalExplanationPlot({ trace, title, xaxistitle }) {
  return (
    <Plot
      data={[trace]}
      style={{}}
      layout={{
        title: title,
        autosize: true,
        yaxis: {
          type: "category",
          automargin: true,
          dtick: 1,
          categoryorder: "total ascending"
        },
        xaxis: {
          title: xaxistitle,
          dtick: 0.1,
          ticks: "inside",
          tickangle: 45
        },
        margin: {
          l: 0,
          r: 40,
          t: 40,
          p: 0
        },
        font: {
          family: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"`,
          size: 16
        }
      }}
      config={{ displayModeBar: false, responsive: true }}
    />
  )
}

export { GlobalExplanationPlot, getTrace, getDifferences, getNames }
