import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Button from "react-bootstrap/Button"

import Octicon, { Graph } from "@primer/octicons-react"
import Rules from "./Rules"
import {
  ExplanationPlot,
  getTrace,
  getDifferences,
  getNames
} from "./ExplanationPlot"

function ExplanationComparison() {
  const [explanation, setExplanation] = useState(null)
  const [explanation2, setExplanation2] = useState(null)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/explanation_comparison")
      const json = await res.json()
      setExplanation(json["exp1"])
      setExplanation2(json["exp2"])
    }

    fetchData()
  }, [])

  if (explanation === null || explanation2 === null) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>Explanation Comparison</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(explanation)

  const names = getNames(explanation)

  const trace = getTrace(differences, names)

  const differences2 = getDifferences(explanation2)

  const names2 = getNames(explanation2)

  const trace2 = getTrace(differences2, names2)

  return (
    <Container>
      <Row className="mt-3 mb-3 d-flex align-items-center">
        <Col>
          <h2>Explanation Comparison</h2>
          <p>
            Explanation of instance <code>{explanation.instance_id}</code> of
            dataset <code> {explanation.explainer_info.dataset_name} </code> for
            the <code> {explanation.explainer_info.classifier_name} </code> and{" "}
            <code> {explanation2.explainer_info.classifier_name} </code>{" "}
            classifiers. It belong to class <b>{explanation.target_class}</b>{" "}
            with probability <code>{explanation.prob.toFixed(3)}</code> for{" "}
            <code> {explanation.explainer_info.classifier_name} </code>{" "}
            classifier and <code>{explanation2.prob.toFixed(3)}</code> for{" "}
            <code> {explanation2.explainer_info.classifier_name} </code>. True
            class: <code> {explanation.true_class} </code>
          </p>
        </Col>
        <Col xs={2}>
          {" "}
          <Button
            variant="outline-dark"
            className="ml-auto p-2"
            href="/analyses_new"
          >
            {" "}
            <Octicon icon={Graph} /> New analyses{" "}
          </Button>{" "}
        </Col>
      </Row>
      <Row>
        <Col xs={6}>
          <ExplanationPlot
            trace={trace}
            title={
              "Dataset: " +
              explanation.explainer_info.dataset_name +
              "  model=" +
              explanation.explainer_info.classifier_name +
              "<br>p(y=" +
              explanation.target_class +
              "|" +
              explanation.explainer_info.meta +
              ")=" +
              explanation.prob.toFixed(3) +
              "  true class=" +
              explanation.true_class
            }
            xaxistitle={"Δ - target class = " + explanation.target_class}
          />
          <Rules explanation={explanation} />
        </Col>
        <Col xs={6}>
          <ExplanationPlot
            trace={trace2}
            title={
              "Dataset: " +
              explanation2.explainer_info.dataset_name +
              "  model=" +
              explanation2.explainer_info.classifier_name +
              "<br>p(y=" +
              explanation2.target_class +
              "|" +
              explanation2.explainer_info.meta +
              ")=" +
              explanation2.prob.toFixed(3) +
              "  true class=" +
              explanation2.true_class
            }
            xaxistitle={"Δ - target class = " + explanation2.target_class}
          />
          <Rules explanation={explanation2} />
        </Col>
      </Row>
    </Container>
  )
}

export default ExplanationComparison
