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

function Analyses() {
  const [analyses, setAnalyses] = useState([])
  const [toGlobalExplanation, setToGlobalExplanation] = useState(false)
  const [toRedirect, setToRedirect] = useState(false)
  const [toMispredicted, setToMispredicted] = useState(false)
  const [toClassComparison, setToClassComparison] = useState(false)

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/analyses")
      const json = await res.json()
      setAnalyses(json)
    }

    fetchData()
  }, [])

  function postAnalysis(analysisName) {
    return async () => {
      if (analysisName === "global_explanation") {
        setToGlobalExplanation(true)
      }
      if (analysisName === '2mispredicted') {
        setToMispredicted(true)
      }
      if (analysisName === '2t_class_comparison') {
        setToClassComparison(true)
      }
      setToRedirect(true)
      await fetch(`http://127.0.0.1:5000/analyses/${analysisName}`, {
        method: "POST"
      })
    }
  }

  if (toRedirect) {
    if (toGlobalExplanation) {
      return <Redirect to="/global_explanation" />
    }
    if (toMispredicted) {
      return <Redirect to="/mispred_instances" />
    }
    if (toClassComparison) {
      return <Redirect to="/instances_class_comparison" />
    }
    return <Redirect to="/instances" />
  }

  return (
    <Container>
      <Row>
        <Col>
          <h2>Select the analysis to perform</h2>
        </Col>
      </Row>
      <Row className="justify-content-md-center">
        <Col lg={6}>
          <ListGroup>
            {Object.entries(analyses).map(([id, { display_name }]) => (
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
    </Container>
  )
}

export default Analyses
