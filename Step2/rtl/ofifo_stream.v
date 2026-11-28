module ofifo_stream #(
    parameter COLS = 8,
    parameter WIDTH = 16,
    parameter DEPTH = 16 // Power of two, at least two words.
)(
    input clk,
    input reset,
    input [COLS-1:0] wr,
    input [COLS*WIDTH-1:0] in,
    input rd,
    output [COLS*WIDTH-1:0] out,
    output o_full,
    output o_valid
);
    localparam ADDR_BITS = $clog2(DEPTH);
    wire [COLS-1:0] full, empty;
    wire pop = rd && o_valid;
    assign o_valid = ~|empty;
    assign o_full = |full;

    genvar c;
    generate for (c = 0; c < COLS; c = c + 1) begin : lane
        reg [WIDTH-1:0] data [0:DEPTH-1];
        reg [ADDR_BITS:0] read_ptr, write_ptr;
        wire [ADDR_BITS-1:0] head = read_ptr[ADDR_BITS-1:0];
        wire [ADDR_BITS-1:0] tail = write_ptr[ADDR_BITS-1:0];

        assign empty[c] = read_ptr == write_ptr;
        assign full[c] = head == tail && read_ptr[ADDR_BITS] != write_ptr[ADDR_BITS];
        assign out[c*WIDTH +: WIDTH] = data[head];

        always @(posedge clk) begin
            if (reset) begin
                read_ptr <= 0;
                write_ptr <= 0;
            end else begin
                if (pop) read_ptr <= read_ptr + 1'b1;
                if (wr[c] && (!full[c] || pop)) begin
                    data[tail] <= in[c*WIDTH +: WIDTH];
                    write_ptr <= write_ptr + 1'b1;
                end
            end
        end
    end endgenerate
endmodule
