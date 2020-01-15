import { useTable, useSortBy, usePagination } from "react-table"
import Table from "react-bootstrap/Table"
import React from "react"
import Col from "react-bootstrap/Col"
import Form from "react-bootstrap/Form"
import Button from "react-bootstrap/Button"
import ButtonGroup from "react-bootstrap/ButtonGroup"

function MyTable({ columns, data, onCheck, isChecked }) {
  // Use the state and functions returned from useTable to build your UI
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    prepareRow,
    page, // Instead of using 'rows', we'll use page,
    // which has only the rows for the active page

    // The rest of these things are super handy, too ;)
    canPreviousPage,
    canNextPage,
    pageOptions,
    pageCount,
    gotoPage,
    nextPage,
    previousPage,
    setPageSize,
    state: { pageIndex, pageSize }
  } = useTable(
    {
      columns,
      data,
      initialState: { pageIndex: 0 }
    },
    useSortBy,
    usePagination
  )

  // Render the UI for your table
  return (
    <>
      <Table
        {...getTableProps()}
        hover
        style={{
          display: "block",
          overflowX: "auto",
          whiteSpace: "nowrap"
        }}
      >
        <thead>
          {headerGroups.map(headerGroup => (
            <tr {...headerGroup.getHeaderGroupProps()}>
              <th>{""}</th>
              {headerGroup.headers.map(column => (
                <th {...column.getHeaderProps(column.getSortByToggleProps())}>
                  {column.render("Header")}
                  <span>
                    {column.isSorted
                      ? column.isSortedDesc
                        ? " 🔽"
                        : " 🔼"
                      : ""}
                  </span>
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody {...getTableBodyProps()}>
          {page.map(row => {
            prepareRow(row)
            return (
              <tr {...row.getRowProps()}>
                <td>
                  <Form.Check
                    checked={isChecked(row)}
                    type="radio"
                    onChange={onCheck(row)}
                  />
                </td>
                {row.cells.map(cell => {
                  return <td {...cell.getCellProps()}>{cell.render("Cell")}</td>
                })}
              </tr>
            )
          })}
        </tbody>
      </Table>
      {(() => {
        if (pageOptions.length > 1) {
          return (
            <div>
              <div>
                <ButtonGroup className={"mr-3"} variant="dark">
                  <Button
                    variant="dark"
                    onClick={() => gotoPage(0)}
                    disabled={!canPreviousPage}
                  >
                    {"<<"}
                  </Button>{" "}
                  <Button
                    variant="dark"
                    onClick={() => previousPage()}
                    disabled={!canPreviousPage}
                  >
                    {"<"}
                  </Button>{" "}
                  <Button
                    variant="dark"
                    onClick={() => nextPage()}
                    disabled={!canNextPage}
                  >
                    {">"}
                  </Button>{" "}
                  <Button
                    variant="dark"
                    onClick={() => gotoPage(pageCount - 1)}
                    disabled={!canNextPage}
                  >
                    {">>"}
                  </Button>
                </ButtonGroup>

                <span>
                  Page{" "}
                  <strong>
                    {pageIndex + 1} of {pageOptions.length}
                  </strong>{" "}
                </span>
              </div>
              <div className={"mt-3 mb-3"}>
                <Form.Row>
                  <Col>
                    <Form.Group>
                      <Form.Control
                        size={"sm"}
                        type="number"
                        defaultValue={pageIndex + 1}
                        onChange={e => {
                          const page = e.target.value
                            ? Number(e.target.value) - 1
                            : 0
                          gotoPage(page)
                        }}
                      />
                    </Form.Group>
                  </Col>
                  <Col>
                    <Form.Group>
                      <Form.Control
                        size={"sm"}
                        as={"select"}
                        value={pageSize}
                        onChange={e => {
                          setPageSize(Number(e.target.value))
                        }}
                      >
                        {[10, 20, 30, 40, 50].map(pageSize => (
                          <option key={pageSize} value={pageSize}>
                            Show {pageSize}
                          </option>
                        ))}
                      </Form.Control>
                    </Form.Group>
                  </Col>
                </Form.Row>
              </div>
            </div>
          )
        }
      })()}
    </>
  )
}

function makeInstances(response) {
  return response.instances.map(instance => {
    const row = {}
    row["id"] = instance[1]
    response.domain.forEach((attribute, attribute_ix) => {
      row[attribute[0]] = attribute[1][instance[0][attribute_ix]]
    })
    return row
  })
}

function makeColumns(domain) {
  return [
    {
      Header: "id",
      accessor: "id"
    },
    ...domain.map(attribute => {
      const name = attribute[0]
      return {
        Header: name,
        accessor: name
      }
    })
  ]
}

function makeClasses(classes) {
  return classes.map(c => {
    return {
      type: c
    }
  })
}

export { MyTable, makeInstances, makeColumns, makeClasses }
