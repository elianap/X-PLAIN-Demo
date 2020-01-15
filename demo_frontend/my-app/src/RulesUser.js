import React from "react"

function RulesUser({ explanation, id_user_rules }) {
  const { map_difference, diff_single, domain, instance } = explanation
  return (
    <>
      {Object.entries(map_difference)
        .map((rule, ix) => [rule, ix])
        .sort(([[, v1]], [[, v2]]) => v1 < v2)
        .map(([[rule, contribution], ix]) => (
          <p key={rule} style={{ fontFamily: "serif", fontSize: "1.1rem" }}>
            <span
              style={{
                background: [
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
                ][(((contribution + 1) / 2) * 10) | 0]
              }}
            >
              Rule {ix + 1}
            </span>
            {(() => {
              if (id_user_rules.map(r => r.join(",")).includes(rule)) {
                return <span> (User Rule) </span>
              }
            })()}{" "}
            ={" "}
            {(() => {
              let attribute_indices = rule.split(",")
              attribute_indices.sort((ix_1, ix_2) => {
                return diff_single[ix_1 - 1] < diff_single[ix_2 - 1]
              })

              return (
                <span>
                  {"{"}
                  {attribute_indices
                    .map(ix => domain[ix - 1][0])
                    .map(
                      attribute_name =>
                        `${attribute_name}=${instance[attribute_name].value}`
                    )
                    .join(", ")}
                  {"}"}
                </span>
              )
            })()}
          </p>
        ))}
    </>
  )
}

export default RulesUser
