import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Button from "react-bootstrap/Button"
import Table from "react-bootstrap/Table"
import Dropdown from "react-bootstrap/Dropdown"
import Octicon, { Graph, Sync } from "@primer/octicons-react"
import ButtonGroup from "react-bootstrap/ButtonGroup"

import Rules from "./Rules"
import {
  ExplanationPlot,
  getTrace,
  getDifferences,
  getNames
} from "./ExplanationPlot"

function WhatIf() {
  const [whatIfExplanation, setwhatIfExplanation] = useState(null)
  const [instanceAttributes, setInstanceAttributes] = useState(null)
  const [recomputeLoading, setRecomputeLoading] = useState(false)
  const [oldInstanceAttributes, setOldInstanceAttributes] = useState(null)
  const [oldWhatIfExplanation, setOldWhatIfExplanation] = useState(null)
  const [cnt_revision, setCnt_revision] = useState(1)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/whatIfExplanation")
      const json = await res.json()
      setwhatIfExplanation(json.explanation)
      console.log(json.explanation)
      setInstanceAttributes(json.attributes)
      setOldWhatIfExplanation(json.explanation)
      setOldInstanceAttributes(json.attributes)
      setCnt_revision(json.cnt_revision)
      console.log("E")
    }

    fetchData()
    console.log("G")
  }, [])

  function handleRestore(e) {
    async function fetchData() {
      setwhatIfExplanation(oldWhatIfExplanation)
      setInstanceAttributes(oldInstanceAttributes)
      setCnt_revision(cnt_revision+1)

    }

    fetchData()
  }

  function handleRecompute(e) {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/whatIfExplanation", {
        method: "post",
        body: JSON.stringify(instanceAttributes)
      })
      const json = await res.json()
      setwhatIfExplanation(json.explanation)
      setInstanceAttributes(json.attributes)
      setRecomputeLoading(false)
      setCnt_revision(json.cnt_revision)
      console.log("G")

    }
    console.log("i")
    setRecomputeLoading(true)
    fetchData()
  }

  if (whatIfExplanation === null || instanceAttributes === null) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>What If Analysis</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(whatIfExplanation)
  const names = getNames(whatIfExplanation)
  const trace = getTrace(differences, names)
  console.log("A", cnt_revision)
  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <Col xs={7}>
          <h2 className="p-2">What If analysis</h2>{" "}
        </Col>
        <Col>
          {recomputeLoading ? (
            <Button variant="dark" disabled>
              <Spinner
                as="span"
                size="sm"
                animation="border"
                role="status"
                aria-hidden="true"
              />
              <span className={"ml-2"}>Recomputing...</span>
              <span className="sr-only">Loading...</span>
            </Button>
          ) : (
            <ButtonGroup>
              <Button
                variant="outline-dark"
                className="ml-auto p-2"
                href="/analyses_new"
              >
                {" "}
                <Octicon icon={Graph} /> New analyses{" "}
              </Button>
              <Button
                variant="secondary"
                className="ml-auto p-2"
                onClick={handleRestore}
              >
                {" "}
                Restore{" "}
              </Button>
              <Button
                variant="dark"
                className="ml-auto p-2"
                onClick={handleRecompute}
              >
                {" "}
                <Octicon icon={Sync} /> Recompute explanation{" "}
              </Button>
            </ButtonGroup>
          )}{" "}
        </Col>
      </Row>

      <Row className="mb-3">
        <Col>
          <Table size="sm">
            <thead>
              <tr>
                <td>Feature</td>
                <td>Values</td>
              </tr>
            </thead>
            <tbody>
              {Object.entries(instanceAttributes).map(
                ([name, { options, value }]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      <Dropdown
                        variant="dark"
                        onSelect={newValue => {
                          const newInstanceAttributes = {
                            ...instanceAttributes
                          }
                          newInstanceAttributes[name] = {
                            ...newInstanceAttributes[name],
                            value: newValue
                          }

                          setInstanceAttributes(newInstanceAttributes)
                        }}
                      >
                        <Dropdown.Toggle variant="dark" id={name}>
                          {value}
                        </Dropdown.Toggle>
                        <Dropdown.Menu>
                          {options.map(o => (
                            <Dropdown.Item eventKey={o} key={name + o}>
                              {o}
                            </Dropdown.Item>
                          ))}
                        </Dropdown.Menu>
                      </Dropdown>
                    </td>
                  </tr>
                )
              )}
            </tbody>
          </Table>
        </Col>
        <Col>
          <ExplanationPlot
            trace={trace}
            title={
              "Dataset: " +
              whatIfExplanation.explainer_info.dataset_name +
              "  model=" +
              whatIfExplanation.explainer_info.classifier_name +
              "<br>p(y=" +
              whatIfExplanation.target_class +
              "|" +
              whatIfExplanation.explainer_info.meta +
              ")=" +
              whatIfExplanation.prob.toFixed(3)
            }
            xaxistitle={"Δ - target class = " + whatIfExplanation.target_class}
            cnt_revision={cnt_revision}
          />
          <p>
            The instance <code>{whatIfExplanation.instance_id}</code> belongs to
            the class <b>{whatIfExplanation.target_class}</b> with probability{" "}
            <code>{whatIfExplanation.prob.toFixed(3)}</code>.
          </p>
          <p>
            The method has converged with error{" "}
            <code>{whatIfExplanation.error.toFixed(3)}</code> and a locality of
            size <code>{whatIfExplanation.k}</code> (parameter K).
          </p>
          <Rules explanation={whatIfExplanation} />
        </Col>
      </Row>
    </Container>
  )
}

export default WhatIf
