import React, { useState, useEffect } from "react"
import Container from "react-bootstrap/Container"
import Row from "react-bootstrap/Row"
import Col from "react-bootstrap/Col"
import Spinner from "react-bootstrap/Spinner"
import Button from "react-bootstrap/Button"
import { makeStyles, useTheme } from "@material-ui/core/styles"
import Input from "@material-ui/core/Input"
import InputLabel from "@material-ui/core/InputLabel"
import MenuItem from "@material-ui/core/MenuItem"
import FormControl from "@material-ui/core/FormControl"
import Select from "@material-ui/core/Select"
import Chip from "@material-ui/core/Chip"

import Octicon, { Graph, Sync } from "@primer/octicons-react"
import ButtonGroup from "react-bootstrap/ButtonGroup"
import RulesUser from "./RulesUser"

import {
  ExplanationPlot,
  getTrace,
  getDifferences,
  getNames
} from "./ExplanationPlot"

function UserRules() {
  const [userRulesExplanation, setuserRulesExplanation] = useState(null)
  const [recomputeLoading, setRecomputeLoading] = useState(false)
  const [names_attributes, setNameAttributes] = useState([])
  const [SelectedAttributes, setSelectedAttributes] = React.useState([])
  const [userRuleIndexes, setUserRuleIndexes] = React.useState([])

  useEffect(() => {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/user_rules")
      const json = await res.json()
      setuserRulesExplanation(json.explanation)
      setNameAttributes(json.attributes)
    }
    fetchData()
  }, [])

  function handleRecompute(e) {
    async function fetchData() {
      const res = await fetch("http://127.0.0.1:5000/user_rules", {
        method: "post",
        body: JSON.stringify(SelectedAttributes)
      })
      const json = await res.json()
      setuserRulesExplanation(json.explanation)
      setUserRuleIndexes(json.id_user_rules)
      setRecomputeLoading(false)
    }

    setRecomputeLoading(true)
    fetchData()
  }

  const useStyles = makeStyles(theme => ({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 120,
      maxWidth: 300
    },
    chips: {
      display: "flex",
      flexWrap: "wrap"
    },
    chip: {
      margin: 2
    },
    noLabel: {
      marginTop: theme.spacing(3)
    }
  }))

  const ITEM_HEIGHT = 48
  const ITEM_PADDING_TOP = 8
  const MenuProps = {
    PaperProps: {
      style: {
        maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
        width: 250
      }
    }
  }

  function getStyles(name, SelectedAttributes, theme) {
    return {
      fontWeight:
        SelectedAttributes.indexOf(name) === -1
          ? theme.typography.fontWeightRegular
          : theme.typography.fontWeightMedium
    }
  }

  const classes = useStyles()
  const theme = useTheme()

  const handleChange = event => {
    setSelectedAttributes(event.target.value)
  }

  if (userRulesExplanation === null || names_attributes === null) {
    return (
      <Container>
        <Row>
          <Col className="mt-3">
            <h2>User Rules</h2>
            <Spinner animation="border" />
          </Col>
        </Row>
      </Container>
    )
  }

  const differences = getDifferences(userRulesExplanation)
  const names = getNames(userRulesExplanation)
  const trace = getTrace(differences, names)

  return (
    <Container>
      <Row className="mt-3 d-flex align-items-center">
        <Col xs={7}>
          <h2 className="p-2">User Rules</h2>
        </Col>
        <Col>
          {recomputeLoading ? (
            <Button className="ml-auto p-2" variant="dark" disabled>
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
                className="ml-auto p-2"
                variant="dark"
                onClick={handleRecompute}
              >
                <Octicon icon={Sync} /> Add Rule and Recompute
              </Button>
            </ButtonGroup>
          )}
        </Col>
      </Row>
      <Row className="mb-3">
        <Col xs={4}>
          <div>
            <FormControl className={classes.formControl}>
              <InputLabel id="demo-mutiple-chip-label">Attributes</InputLabel>
              <Select
                labelId="demo-mutiple-chip-label"
                id="demo-mutiple-chip"
                multiple
                value={SelectedAttributes}
                onChange={handleChange}
                input={<Input id="select-multiple-chip" />}
                renderValue={selected => (
                  <div className={classes.chips}>
                    {selected.map(value => (
                      <Chip
                        key={value}
                        label={value}
                        className={classes.chip}
                      />
                    ))}
                  </div>
                )}
                MenuProps={MenuProps}
              >
                {names_attributes.map(name => (
                  <MenuItem
                    key={name}
                    value={name}
                    style={getStyles(name, SelectedAttributes, theme)}
                  >
                    {name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </div>
        </Col>
        <Col>
          <ExplanationPlot
            trace={trace}
            title={
              "Dataset: " +
              userRulesExplanation.explainer_info.dataset_name +
              "  model=" +
              userRulesExplanation.explainer_info.classifier_name +
              "<br>p(y=" +
              userRulesExplanation.target_class +
              "|" +
              userRulesExplanation.explainer_info.meta +
              ")=" +
              userRulesExplanation.prob.toFixed(3) +
              "  true class=" +
              userRulesExplanation.true_class
            }
            xaxistitle={
              "Δ - target class = " + userRulesExplanation.target_class
            }
          />
          <p>
            The instance <code>{userRulesExplanation.instance_id}</code> belongs
            to the class <b>{userRulesExplanation.target_class}</b> with
            probability <code>{userRulesExplanation.prob.toFixed(3)}</code>.
          </p>
          <p>
            The method has converged with error{" "}
            <code>{userRulesExplanation.error.toFixed(3)}</code> and a locality
            of size <code>{userRulesExplanation.k}</code> (parameter K).
          </p>
          <RulesUser
            explanation={userRulesExplanation}
            id_user_rules={userRuleIndexes}
          />
        </Col>
      </Row>
    </Container>
  )
}

export default UserRules
