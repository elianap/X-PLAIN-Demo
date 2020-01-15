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

function getNames(explanation) {
  return explanation.domain
    .map(([name]) => `${name}=${explanation.instance[name].value}`)
    .concat(
      Object.keys(explanation.map_difference)
        .map(function(rule, ix) {
          if (rule.length === 1) {
            return null
          } else {
            return `Rule ${ix + 1}`
          }
        })
        .filter(x => x != null)
    )
}

function getDifferences(explanation) {
  return explanation.diff_single.concat(
    Object.keys(explanation.map_difference)
      .map(function(key, _) {
        if (key.length === 1) {
          return null
        } else {
          return explanation.map_difference[key]
        }
      })
      .filter(x => x != null)
  )
}

//    {compare ? '' : '{{}}'}
//style={{width: '100%', height: '100%'}}
function ExplanationPlot({ trace, title, xaxistitle, cnt_revision}) {
  return (
    <Plot
      data={[trace]}
      style={{}}
      layout={{
        title: {
          text: title,
          font: {
            size: 14
          }
        },
        autosize: true,
        yaxis: {
          type: "category",
          automargin: true,
          dtick: 1,
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

export { ExplanationPlot, getTrace, getDifferences, getNames }


//          categoryorder: "total ascending"
