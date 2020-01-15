import React, { useState, useEffect } from "react"
import { Redirect } from "react-router-dom"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import ListGroup from "react-bootstrap/ListGroup"
import Octicon, {
  Question,
  MortarBoard,
  PrimitiveDot,
  Search,
  Person,
  Law,
  Globe,
  Versions
} from "@primer/octicons-react"

function AnalysesNew() {
  const [analysesInstance, setAnalysesInstance] = useState([])
  const [analysesGlobal, setAnalysesGlobal] = useState([])

  const [toExplanation, setToExplanation] = useState(false)
  const [toWhatIf, setToWhatIf] = useState(false)
  const [toMispredicted, setToMispredicted] = useState(false)
  const [toUserRules, setToUserRules] = useState(false)
  const [toExplanationComparison, setToExplanationComparison] = useState(false)
  const [toGlobalExplanation, setToGlobalExplanation] = useState(false)
  const [response, setResponse] = useState(false)
  const [toClassComparison, setToClassComparison] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/analyses_new")
      const json = await res.json()
      setResponse(json)
      setAnalysesInstance(json.analyses_on_instance)
      setAnalysesGlobal(json.global_analyses)
    }

    fetchData()
  }, [])

  function postAnalysis(analysisName) {
    var proceed = "false"
    return async () => {
      if (analysisName === '1explain') {
        setToExplanation(true)
      }
      if (analysisName === '4whatif') {
        setToWhatIf(true)
        proceed = "true"
      }
      if (analysisName === '2mispredicted') {
        setToMispredicted(true)
      }
      if (analysisName === '3user_rules') {
        setToUserRules(true)
        proceed = "true"
      }
      if (analysisName === "3explaination_comparison") {
        setToExplanationComparison(true)
        proceed = "true"
      }
      if (analysisName === "global_explanation") {
        setToGlobalExplanation(true)
      }
      if (analysisName === '2t_class_comparison') {
        setToClassComparison(true)
        proceed = "true"
      }
      await fetch(`http://127.0.0.1:5000/analyses_new/${proceed}`, {
        method: "POST"
      })
    }
  }

  if (toExplanation) {
    return <Redirect to="/instances" />
  }
  if (toWhatIf) {
    return <Redirect to="/whatif" />
  }

  if (toMispredicted) {
    return <Redirect to="/mispred_instances" />
  }

  if (toUserRules) {
    return <Redirect to="/user_rules" />
  }
  if (toExplanationComparison) {
    return <Redirect to="/classifiers_2" />
  }

  if (toGlobalExplanation) {
    return <Redirect to="/global_explanation" />
  }
  if (toClassComparison) {
    return <Redirect to="/class_comparison" />
  }

  return (
    <Container>
      <Row>
        <Col>
          <h2>Select the analysis to perform</h2>
        </Col>
      </Row>
      <Row>
        <p>
          Continue the analysis of instance <code>{response.instance_id}</code>{" "}
          of dataset <code>{response.dataset_name}</code> for the{" "}
          <code>{response.classifier_name}</code> classifier
        </p>
      </Row>
      <Row className="justify-content-md-center">
        <Col lg={6}>
          <ListGroup>
            {Object.entries(analysesInstance).map(([id, { display_name }]) => (
              <ListGroup.Item
                variant="info"
                className="text-center"
                action
                key={id}
                onClick={postAnalysis(id)}
              >
                <Octicon
                  icon={(id => {
                    switch (id) {
                      case '1explain':
                        return Question

                      case '4whatif':
                        return MortarBoard

                      case '2mispredicted':
                        return Search

                      case '3user_rules':
                        return Person

                      case "3explaination_comparison":
                        return Law

                      case "global_explanation":
                        return Globe

                      case '2t_class_comparison':
                        return Versions

                      default:
                        return PrimitiveDot
                    }
                  })(id)}
                />{" "}
                {display_name}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
      <Row>
        {" "}
        <p>Proceed with new analysis</p>
      </Row>
      <Row className="justify-content-md-center">
        <Col lg={6}>
          <ListGroup>
            {Object.entries(analysesGlobal).map(([id, { display_name }]) => (
              <ListGroup.Item
                className="text-center"
                action
                key={id}
                onClick={postAnalysis(id)}
              >
                <Octicon
                  icon={(id => {
                    switch (id) {
                      case '1explain':
                        return Question

                      case '4whatif':
                        return MortarBoard

                      case '2mispredicted':
                        return Search

                      case '3user_rules':
                        return Person

                      case "3explaination_comparison":
                        return Law

                      case "global_explanation":
                        return Globe

                      default:
                        return PrimitiveDot
                    }
                  })(id)}
                />{" "}
                {display_name}
              </ListGroup.Item>
            ))}
          </ListGroup>
        </Col>
      </Row>
    </Container>
  )
}

export default AnalysesNew
